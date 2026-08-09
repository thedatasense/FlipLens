"""Train a small sparse autoencoder (SAE) on the nanoVLM residual stream and test
whether an UNSUPERVISED feature rediscovers the paraphrase-flip direction that
Experiment E found by supervised difference-of-means, with a PCA baseline.

If an SAE feature aligns with the Experiment E flip direction and predicts flips,
the low-rank direction can be called a feature found the same unsupervised way
GemmaScope found Feature 3818 on the real model.

    python sae.py --layer 1 --out results/sae

See docs/tiny_vlm_psf_isolation.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.set_num_threads(4)
from babygemma.paths import ROOT as HERE

from babygemma import metrics as Mx
from babygemma.dataset import collate
from babygemma.training import train_model


# ---- capture residual activations at the ANS position, per layer ----
@torch.no_grad()
def capture(model, dataset, device, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    per_layer = defaultdict(list)
    meta = []  # (cluster_id, phenomenon, pred)
    for b in loader:
        idx = (b["ans_pos"] + model.n_img).to(device)
        kw = {"ground": b["ground"].to(device)} if getattr(model, "use_ground", False) else {}
        logits, acts = model(b["vision"].to(device), b["tokens"].to(device),
                             b["ans_pos"].to(device), capture=True, **kw)
        preds = logits.argmax(-1).cpu().tolist()
        B = b["tokens"].shape[0]
        ar = torch.arange(B, device=device)
        for L, h in enumerate(acts):
            per_layer[L].append(h[ar, idx].cpu())
        for j in range(B):
            meta.append((int(b["cluster_id"][j]), b["phenomenon"][j], preds[j]))
    per_layer = {L: torch.cat(v).float() for L, v in per_layer.items()}
    return per_layer, meta


# ---- top-k sparse autoencoder ----
class TopKSAE(nn.Module):
    def __init__(self, d, m, k):
        super().__init__()
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d, bias=False)

    def forward(self, x):
        z = self.enc(x - self.b_pre)
        topv, topi = z.topk(self.k, dim=-1)
        zk = torch.zeros_like(z).scatter_(-1, topi, F.relu(topv))
        return self.dec(zk) + self.b_pre, zk

    @torch.no_grad()
    def normalize_decoder(self):
        self.dec.weight.data = F.normalize(self.dec.weight.data, dim=0)


def train_sae(acts, d, m=2048, k=16, steps=3000, lr=1e-3, device="cuda"):
    X = acts.to(device)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    sae = TopKSAE(d, m, k).to(device)
    sae.b_pre.data = Xn.mean(0).clone()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = Xn.shape[0]
    for step in range(steps):
        gidx = torch.randint(0, n, (512,), device=device)
        recon, _ = sae(Xn[gidx])
        loss = F.mse_loss(recon, Xn[gidx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            sae.normalize_decoder()
    sae.eval()
    with torch.no_grad():
        _, Z = sae(Xn)
        var_expl = 1 - (F.mse_loss(sae(Xn)[0], Xn) / Xn.var())
    return sae, Z.cpu(), (mu.cpu(), sd.cpu()), float(var_expl)


def flip_direction(acts_layer, meta, flips, device):
    """Experiment E direction at this layer: mean(donor - target) over flipped
    clusters (donor = majority-pred example, target = a minority-pred example)."""
    by_cluster = defaultdict(list)
    for i, (c, ph, pr) in enumerate(meta):
        by_cluster[c].append((i, pr))
    diffs = []
    for c, items in by_cluster.items():
        if not flips.get(c):
            continue
        preds = [pr for _, pr in items]
        maj = Counter(preds).most_common(1)[0][0]
        donor = next(i for i, pr in items if pr == maj)
        target = next(i for i, pr in items if pr != maj)
        diffs.append((acts_layer[donor] - acts_layer[target]).numpy())
    if not diffs:
        return None
    d = np.mean(diffs, axis=0)
    return d / (np.linalg.norm(d) + 1e-8)


def score_features(Z, meta, flips, feat_dirs, flip_dir):
    """Per feature: (a) cosine of its dictionary direction with the E flip
    direction; (b) point-biserial of per-cluster mean activation with flip."""
    # per-cluster mean feature activation
    by_cluster = defaultdict(list)
    for i, (c, ph, pr) in enumerate(meta):
        by_cluster[c].append(i)
    cids = [c for c in by_cluster if c in flips]
    y = np.array([1.0 if flips[c] else 0.0 for c in cids])
    Znp = Z.numpy()
    cluster_mean = np.stack([Znp[by_cluster[c]].mean(0) for c in cids])  # [C, m]

    cos = feat_dirs @ flip_dir if flip_dir is not None else np.zeros(feat_dirs.shape[0])
    # point-biserial per feature (guard zero-variance)
    pb = np.zeros(cluster_mean.shape[1])
    if len(set(y.tolist())) == 2:
        ys = (y - y.mean())
        for f in range(cluster_mean.shape[1]):
            x = cluster_mean[:, f]
            if x.std() > 1e-8:
                pb[f] = np.corrcoef(x, y)[0, 1]
    return cos, pb


def register_selectivity(Z, meta, feature):
    """Mean activation of one feature per register (phenomenon)."""
    by_ph = defaultdict(list)
    Znp = Z.numpy()
    for i, (c, ph, pr) in enumerate(meta):
        by_ph[ph].append(Znp[i, feature])
    return {ph: float(np.mean(v)) for ph, v in by_ph.items()}


def run(layer, out, seed=0, m=2048, k=16, use_ground=False, model_steps=1500):
    # the probe being analysed must be converged; the scaled model needs ~8k steps,
    # so analysing it at the old 1,500-step default would describe an undertrained net
    art = train_model(regime="augmented", seed=seed,
                      use_ground=use_ground, steps=model_steps)
    model, ds, device = art["model"], art["eval_ds"], art["device"]
    print(f"[sae] model acc={art['result']['accuracy']:.3f} flip={art['result']['flip_rate']:.3f}")

    per_layer, meta = capture(model, ds, device)
    preds = [pr for _, _, pr in meta]
    clusters = [c for c, _, _ in meta]
    flips = Mx.flip_labels(preds, clusters)
    print(f"[sae] captured {len(meta)} examples, {sum(flips.values())} flipped clusters")

    acts = per_layer[layer]
    d = acts.shape[1]
    sae, Z, (mu, sd), var_expl = train_sae(acts, d, m=m, k=k, device=device)
    feat_dirs = F.normalize(sae.dec.weight.detach().cpu(), dim=0).T.numpy()  # [m, d]
    fdir = flip_direction(acts, meta, flips, device)

    cos, pb = score_features(Z, meta, flips, feat_dirs, fdir)

    # PCA baseline
    Xn = ((acts - mu) / sd).numpy()
    Xc = Xn - Xn.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca_dirs = Vt[:20]  # top 20 PCs [20, d]
    pca_cos = np.abs(pca_dirs @ fdir) if fdir is not None else np.zeros(20)

    best_cos = int(np.argmax(np.abs(cos)))
    best_pb = int(np.argmax(np.abs(pb)))
    rand_cos = float(np.mean([abs(np.dot(np.random.randn(d) / np.sqrt(d), fdir)) for _ in range(200)])) if fdir is not None else 0.0

    result = {
        "layer": layer, "sae_dict": m, "topk": k, "var_explained": var_expl,
        "n_examples": len(meta), "n_flip_clusters": int(sum(flips.values())),
        "sae_feature_max_cos_with_E_dir": {"feature": best_cos, "cos": float(cos[best_cos]),
                                           "flip_pointbiserial": float(pb[best_cos])},
        "sae_feature_max_flip_pb": {"feature": best_pb, "pb": float(pb[best_pb]),
                                    "cos_with_E_dir": float(cos[best_pb])},
        "pca_top20_max_cos_with_E_dir": float(np.max(pca_cos)),
        "random_dir_mean_cos_with_E_dir": rand_cos,
        "register_selectivity_of_top_flip_feature": register_selectivity(Z, meta, best_pb),
    }
    print("\n[sae] RESULTS")
    print(f"  variance explained by SAE: {var_expl:.3f}")
    print(f"  best SAE feature by |cos with E flip direction|: feature {best_cos}, "
          f"cos={cos[best_cos]:+.3f}, flip point-biserial={pb[best_cos]:+.3f}")
    print(f"  best SAE feature by flip point-biserial: feature {best_pb}, "
          f"pb={pb[best_pb]:+.3f}, cos with E dir={cos[best_pb]:+.3f}")
    print(f"  PCA top-20 max |cos with E dir|: {np.max(pca_cos):.3f}")
    print(f"  random direction mean |cos with E dir|: {rand_cos:.3f}")
    print(f"  register selectivity of top flip feature: "
          f"{ {kk: round(vv,3) for kk,vv in result['register_selectivity_of_top_flip_feature'].items()} }")

    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(result, open(os.path.join(out, "sae.json"), "w"), indent=2)
        print(f"[sae] saved {out}/sae.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dict", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--out", default="results/sae")
    ap.add_argument("--arch", default="gemma", choices=["gemma"])  # kept for run/ compatibility
    ap.add_argument("--grounding-token", action="store_true")
    ap.add_argument("--model-steps", type=int, default=1500)
    a = ap.parse_args()
    run(a.layer, a.out, seed=a.seed, m=a.dict, k=a.topk,
        use_ground=a.grounding_token, model_steps=a.model_steps)


if __name__ == "__main__":
    main()
