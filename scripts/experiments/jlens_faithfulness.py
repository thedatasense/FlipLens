"""Does the Jacobian lens actually predict what the model does?

jlens.py reports where paraphrases diverge, but never checks that its readout tracks
the model. Two measurements close that gap, and neither touches a published number.

1. FIDELITY. Per layer, correlate the lens margin with the true yes-minus-no margin
   from the same forward pass. read_lens already computes the logits and throws the
   margin away. This is the scalar analogue of the perplexity check the tuned lens
   uses to earn its claims.

2. THE PATCH VALIDATION. Differencing the lens across a donor and a target gives
   exactly the attribution-patching estimate of what experiment C's rank-1 patch will
   do:  predicted = J_l . (h_donor - h_target). Experiment C already runs that patch
   and keeps only the argmax. Keep the margin instead and you can regress one on the
   other. Syed et al. report R^2 = 0.27 and slope 0.531 for attribution patching
   against activation patching; this is the same plot on this model.

    NANO_INDEX=data/index_padchest_balanced_full.json \
    python scripts/experiments/jlens_faithfulness.py --grounding-token \
        --out results_audit/jlens_faithfulness
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from babygemma import metrics as Mx
from babygemma.training import train_model

from jlens import fit_jacobian, read_lens


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _one(ds, i, device):
    it = ds[i]
    return (it["vision"].unsqueeze(0).to(device),
            it["tokens"].unsqueeze(0).to(device),
            torch.tensor([it["ans_pos"]], device=device),
            it["ground"].unsqueeze(0).to(device))


@torch.no_grad()
def patched_margin(model, vis, tok, ap, layer, donor_vec, ai_t, basis, ground=None):
    """The yes-minus-no margin after writing donor_vec into the answer position."""
    seq = model.n_img + model.cfg.max_len
    donor = torch.zeros(1, seq, model.cfg.dim, device=vis.device)
    donor[0, ai_t] = donor_vec
    spec = {"layer": layer, "donor": donor, "positions": "ans", "basis": basis}
    kw = {"ground": ground} if getattr(model, "use_ground", False) else {}
    logits, _ = model(vis, tok, ap, patch=spec, **kw)
    return float(logits[0, 1] - logits[0, 0])


def run(seed, out, n_fit=256, use_ground=False, model_steps=1500, max_pairs=60):
    art = train_model(regime="augmented", seed=seed, use_ground=use_ground,
                      steps=model_steps)
    model, ds, device = art["model"], art["eval_ds"], art["device"]
    model.eval()
    depth = model.cfg.depth

    J, Hbar, mbar, _ = fit_jacobian(model, ds, device, n_fit=n_fit)
    rows, _ = read_lens(model, ds, device, J, Hbar, mbar)

    # ---- 1. fidelity: lens margin against the true margin, per layer -------------
    from torch.utils.data import DataLoader
    from babygemma.dataset import collate
    true = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=256, collate_fn=collate):
            kw = {"ground": b["ground"].to(device)} if getattr(model, "use_ground", False) else {}
            logits, _ = model(b["vision"].to(device), b["tokens"].to(device),
                              b["ans_pos"].to(device), **kw)
            true.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
    true = np.array(true)
    lens = np.stack([r[3] for r in rows])                     # [N, depth]
    fidelity = [pearson(lens[:, L], true) for L in range(depth)]
    r2 = [float(1 - ((true - lens[:, L]) ** 2).sum() / ((true - true.mean()) ** 2).sum())
          for L in range(depth)]
    print("[faith] lens-vs-true margin, Pearson r by layer: "
          f"{[round(x, 3) for x in fidelity]}")
    print(f"[faith] ... as R^2 of the raw lens value:        {[round(x, 3) for x in r2]}")

    # ---- 2. the patch validation -------------------------------------------------
    preds, _, clusters, _ = Mx.predict(model, ds, device)
    idx_by_c = defaultdict(list)
    for i, c in enumerate(clusters):
        idx_by_c[c].append(i)

    pred_delta = [[] for _ in range(depth)]
    meas_delta = [[] for _ in range(depth)]
    n_pairs = 0
    with torch.no_grad():
        for c, idxs in idx_by_c.items():
            cps = [preds[i] for i in idxs]
            if len(set(cps)) < 2 or n_pairs >= max_pairs:
                continue
            maj = Counter(cps).most_common(1)[0][0]
            di = next(i for i in idxs if preds[i] == maj)
            ti = next(i for i in idxs if preds[i] != maj)
            vd, td, ad, gd = _one(ds, di, device)
            vt, tt, at, gt = _one(ds, ti, device)
            kwd = {"ground": gd} if use_ground else {}
            kwt = {"ground": gt} if use_ground else {}
            _, acts_d = model(vd, td, ad, capture=True, **kwd)
            logits_t, acts_t = model(vt, tt, at, capture=True, **kwt)
            base = float(logits_t[0, 1] - logits_t[0, 0])
            ai_d = int(ad.item()) + model.n_img
            ai_t = int(at.item()) + model.n_img
            for L in range(depth):
                dvec, tvec = acts_d[L][0, ai_d], acts_t[L][0, ai_t]
                diff = dvec - tvec
                if diff.norm() <= 1e-6:
                    continue
                b = (diff / diff.norm()).unsqueeze(0)
                got = patched_margin(model, vt, tt, at, L, dvec, ai_t, b, ground=gt)
                pred_delta[L].append(float((J[L] * diff).sum()))
                meas_delta[L].append(got - base)
            n_pairs += 1

    valid = []
    print(f"[faith] patch validation on {n_pairs} donor/target pairs")
    print(f"  {'layer':>5s} {'n':>5s} {'R^2':>8s} {'slope':>8s} {'pearson':>8s}")
    for L in range(depth):
        x, y = np.array(pred_delta[L]), np.array(meas_delta[L])
        if len(x) < 3 or x.std() < 1e-12:
            valid.append({"layer": L, "n": int(len(x))})
            continue
        slope, icpt = np.polyfit(x, y, 1)
        r = pearson(x, y)
        valid.append({"layer": L, "n": int(len(x)), "r": r, "r2": float(r * r),
                      "slope": float(slope), "intercept": float(icpt)})
        print(f"  {L:5d} {len(x):5d} {r * r:8.3f} {slope:8.3f} {r:8.3f}")

    result = {**art["result"], "n_fit": n_fit, "n_pairs": n_pairs,
              "lens_vs_true_pearson_by_layer": fidelity,
              "lens_vs_true_r2_by_layer": r2,
              "patch_validation_by_layer": valid,
              "benchmark_note": "Syed et al. 2310.10348 report R^2 = 0.27, slope 0.531 "
                                "for attribution patching against activation patching"}
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "jlens_faithfulness.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[faith] saved {out}/jlens_faithfulness.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-fit", type=int, default=256)
    ap.add_argument("--model-steps", type=int, default=1500)
    ap.add_argument("--max-pairs", type=int, default=60)
    ap.add_argument("--grounding-token", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.seed, a.out, n_fit=a.n_fit, use_ground=a.grounding_token,
        model_steps=a.model_steps, max_pairs=a.max_pairs)


if __name__ == "__main__":
    main()
