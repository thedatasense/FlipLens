"""Jacobian lens for baby-MedGemma.

A per-layer readout of the yes/no margin that each decoder layer's answer-position
activation is disposed to produce, fit as the average input-output Jacobian
J_l = E[d margin / d h_l] (the same idea as the Jacobian lens used on MedGemma-4B).
We then read the per-layer lens margin for every phrasing of a question and show
where the phrasings diverge, corroborating the causal locus from experiment_e.

    python jlens.py --arch gemma --out results/jlens

The lens is corroboration only; the causal patch (experiment_e) is primary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

torch.set_num_threads(4)
from babygemma.paths import ROOT as HERE

from babygemma import metrics as Mx
from babygemma.dataset import collate
from babygemma.training import train_model


def _layers(model):
    # gemma decoder layers (baby-Gemma) or the plain blocks (nano)
    return model.gemma.layers if hasattr(model, "gemma") else model.blocks


def fit_jacobian(model, dataset, device, n_fit=256):
    """Average Jacobian of the yes/no margin w.r.t. each layer's answer-position
    activation, plus the mean activation and mean margin (for a first-order read)."""
    layers = _layers(model)
    depth = len(layers)
    dim = model.cfg.dim
    Jsum = [torch.zeros(dim, device=device) for _ in range(depth)]
    Hsum = [torch.zeros(dim, device=device) for _ in range(depth)]
    msum, n = 0.0, 0

    captured = {}
    handles = []
    for L in range(depth):
        def mk(L):
            def hook(mod, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                hs.retain_grad()
                captured[L] = hs
            return hook
        handles.append(layers[L].register_forward_hook(mk(L)))

    model.eval()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    for b in loader:
        if n >= n_fit:
            break
        model.zero_grad(set_to_none=True)
        captured.clear()
        vis = b["vision"].to(device)
        kw = {"ground": b["ground"].to(device)} if getattr(model, "use_ground", False) else {}
        logits, _ = model(vis, b["tokens"].to(device), b["ans_pos"].to(device), **kw)
        margin = logits[0, 1] - logits[0, 0]
        margin.backward()
        ai = int(b["ans_pos"][0]) + model.n_img
        for L in range(depth):
            h = captured[L]
            Jsum[L] += h.grad[0, ai].detach()
            Hsum[L] += h[0, ai].detach()
        msum += float(margin.detach())
        n += 1
    for h in handles:
        h.remove()
    J = [Jsum[L] / n for L in range(depth)]
    Hbar = [Hsum[L] / n for L in range(depth)]
    mbar = msum / n
    return J, Hbar, mbar, depth


@torch.no_grad()
def read_lens(model, dataset, device, J, Hbar, mbar):
    """Per-layer lens margin for every example, first-order: mbar + J_l . (h_l - Hbar_l)."""
    layers = _layers(model)
    depth = len(layers)
    captured = {}
    handles = []
    for L in range(depth):
        def mk(L):
            def hook(mod, inp, out):
                captured[L] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook
        handles.append(layers[L].register_forward_hook(mk(L)))
    model.eval()
    loader = DataLoader(dataset, batch_size=128, collate_fn=collate)
    rows = []  # per example: (cluster, phenomenon, pred, [lens margin per layer])
    for b in loader:
        captured.clear()
        kw = {"ground": b["ground"].to(device)} if getattr(model, "use_ground", False) else {}
        logits, _ = model(b["vision"].to(device), b["tokens"].to(device),
                          b["ans_pos"].to(device), **kw)
        preds = logits.argmax(-1).cpu().tolist()
        B = b["tokens"].shape[0]
        ai = (b["ans_pos"] + model.n_img).to(device)
        per_layer = []
        for L in range(depth):
            h = captured[L][torch.arange(B, device=device), ai]     # [B, dim]
            lm = mbar + ((h - Hbar[L]) * J[L]).sum(-1)              # [B]
            per_layer.append(lm.cpu().numpy())
        per_layer = np.stack(per_layer, axis=1)                     # [B, depth]
        for j in range(B):
            rows.append((int(b["cluster_id"][j]), b["phenomenon"][j], preds[j], per_layer[j]))
    for h in handles:
        h.remove()
    return rows, depth


def analyze(rows, depth, flips):
    """Per-layer within-cluster lens-margin divergence, and its coupling to flips."""
    by_cluster = defaultdict(list)
    for c, ph, pr, lm in rows:
        by_cluster[c].append(lm)
    # within-cluster std of the lens margin at each layer
    div = np.zeros(depth)
    div_flip = np.zeros(depth)
    div_noflip = np.zeros(depth)
    nf = nn = 0
    per_layer_pb = np.zeros(depth)
    cluster_div = {c: np.stack(v).std(0) for c, v in by_cluster.items()}  # [depth] per cluster
    for c, d in cluster_div.items():
        div += d
        if flips.get(c):
            div_flip += d; nf += 1
        else:
            div_noflip += d; nn += 1
    div /= len(cluster_div)
    div_flip = div_flip / nf if nf else div_flip
    div_noflip = div_noflip / nn if nn else div_noflip
    # point-biserial: does per-cluster divergence at layer L predict the flip?
    cids = [c for c in cluster_div if c in flips]
    y = np.array([1.0 if flips[c] else 0.0 for c in cids])
    for L in range(depth):
        x = np.array([cluster_div[c][L] for c in cids])
        if len(set(y.tolist())) == 2 and x.std() > 1e-9:
            per_layer_pb[L] = np.corrcoef(x, y)[0, 1]
    return div, div_flip, div_noflip, per_layer_pb


def run(seed, arch, out, n_fit=256, use_ground=False, model_steps=1500):
    art = train_model(regime="augmented", seed=seed,
                      use_ground=use_ground, steps=model_steps)
    model, ds, device = art["model"], art["eval_ds"], art["device"]
    print(f"[jlens] model acc={art['result']['accuracy']:.3f} flip={art['result']['flip_rate']:.3f}")

    preds, _, clusters, _ = Mx.predict(model, ds, device)
    flips = Mx.flip_labels(preds, clusters)

    J, Hbar, mbar, depth = fit_jacobian(model, ds, device, n_fit=n_fit)
    rows, _ = read_lens(model, ds, device, J, Hbar, mbar)
    div, div_flip, div_noflip, pb = analyze(rows, depth, flips)

    # the lens "commit": the shallowest layer where flipping clusters' lens margins
    # diverge markedly more than non-flipping ones
    ratio = [(div_flip[L] / (div_noflip[L] + 1e-9)) for L in range(depth)]
    commit = next((L for L in range(depth) if div_flip[L] > 2 * div_noflip[L] + 1e-6), None)

    result = {
        **art["result"], "arch": arch, "n_fit": n_fit,
        "lens_divergence_by_layer": div.tolist(),
        "lens_divergence_flip_clusters": div_flip.tolist(),
        "lens_divergence_nonflip_clusters": div_noflip.tolist(),
        "flip_vs_nonflip_divergence_ratio": ratio,
        "divergence_flip_pointbiserial_by_layer": pb.tolist(),
        "lens_commit_layer": commit,
    }
    print("[jlens] per-layer lens-margin divergence (flip clusters):",
          [round(x, 3) for x in div_flip])
    print("[jlens] divergence flip / non-flip ratio:", [round(x, 2) for x in ratio])
    print("[jlens] divergence-vs-flip point-biserial:", [round(x, 2) for x in pb])
    print(f"[jlens] lens commit layer (flip divergence > 2x non-flip): {commit}")
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(result, open(os.path.join(out, "jlens.json"), "w"), indent=2)
        print(f"[jlens] saved {out}/jlens.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="gemma", choices=["gemma"])  # kept for run/ compatibility
    ap.add_argument("--grounding-token", action="store_true")
    ap.add_argument("--model-steps", type=int, default=1500)
    ap.add_argument("--n-fit", type=int, default=256)
    ap.add_argument("--out", default="results/jlens")
    a = ap.parse_args()
    run(a.seed, a.arch, a.out, n_fit=a.n_fit,
        use_ground=a.grounding_token, model_steps=a.model_steps)


if __name__ == "__main__":
    main()
