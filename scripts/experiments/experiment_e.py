"""Experiment E (nanoVLM): causal patching on real chest X-rays.

Rank-1 difference-direction ANS-position patch from a donor (majority-answer)
phrasing into a target (flipped) phrasing, per layer, with a random-direction
control and a non-flip-cluster disruption control. Locates where the flip is
decided in the trainable stack above frozen MedSigLIP.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

from babygemma.paths import ROOT as HERE

from babygemma import metrics as Mx
from babygemma.training import train_model


def _one(ds, i, device):
    it = ds[i]
    return (it["vision"].unsqueeze(0).to(device),
            it["tokens"].unsqueeze(0).to(device),
            torch.tensor([it["ans_pos"]], device=device),
            it["ground"].unsqueeze(0).to(device))


@torch.no_grad()
def _patch_pred(model, vis, tok, ap, layer, donor_vec, ai_t, basis=None, ground=None):
    seq = model.n_img + model.cfg.max_len
    donor = torch.zeros(1, seq, model.cfg.dim, device=vis.device)
    donor[0, ai_t] = donor_vec
    spec = {"layer": layer, "donor": donor, "positions": "ans"}
    if basis is not None:
        spec["basis"] = basis
    kw = {"ground": ground} if getattr(model, "use_ground", False) else {}
    logits, _ = model(vis, tok, ap, patch=spec, **kw)
    return int(logits.argmax(-1))


def run(steps, seed, regime, max_clusters, out, use_ground=False):
    art = train_model(regime=regime, seed=seed, steps=steps, use_ground=use_ground)
    model, ds, device = art["model"], art["eval_ds"], art["device"]
    model.eval()
    depth = model.cfg.depth

    preds, _, clusters, _ = Mx.predict(model, ds, device)
    idx_by_c = defaultdict(list)
    for i, c in enumerate(clusters):
        idx_by_c[c].append(i)

    rec = np.zeros(depth); ctl = np.zeros(depth); ntar = 0
    disr = np.zeros(depth); nnf = 0

    with torch.no_grad():
        for c, idxs in idx_by_c.items():
            cps = [preds[i] for i in idxs]
            if len(set(cps)) >= 2 and ntar < max_clusters:
                maj = Counter(cps).most_common(1)[0][0]
                di = next(i for i in idxs if preds[i] == maj)
                ti = next(i for i in idxs if preds[i] != maj)
                vd, td, ad, gd = _one(ds, di, device)
                vt, tt, at, gt = _one(ds, ti, device)
                kwd = {"ground": gd} if use_ground else {}
                kwt = {"ground": gt} if use_ground else {}
                _, acts_d = model(vd, td, ad, capture=True, **kwd)
                _, acts_t = model(vt, tt, at, capture=True, **kwt)
                ai_d = int(ad.item()) + model.n_img
                ai_t = int(at.item()) + model.n_img
                for L in range(depth):
                    dvec = acts_d[L][0, ai_d]
                    tvec = acts_t[L][0, ai_t]
                    diff = dvec - tvec
                    if diff.norm() > 1e-6:
                        b = (diff / diff.norm()).unsqueeze(0)
                        if _patch_pred(model, vt, tt, at, L, dvec, ai_t, basis=b, ground=gt) == maj:
                            rec[L] += 1
                    rb = torch.randn(model.cfg.dim, device=device)
                    rb = (rb / rb.norm()).unsqueeze(0)
                    if _patch_pred(model, vt, tt, at, L, dvec, ai_t, basis=rb, ground=gt) == maj:
                        ctl[L] += 1
                ntar += 1
            elif len(set(cps)) == 1 and nnf < max_clusters and len(idxs) >= 2:
                same = cps[0]
                vd, td, ad, gd = _one(ds, idxs[0], device)
                vt, tt, at, gt = _one(ds, idxs[1], device)
                _, acts_d = model(vd, td, ad, capture=True,
                                  **({"ground": gd} if use_ground else {}))
                ai_d = int(ad.item()) + model.n_img
                ai_t = int(at.item()) + model.n_img
                for L in range(depth):
                    if _patch_pred(model, vt, tt, at, L, acts_d[L][0, ai_d], ai_t, ground=gt) != same:
                        disr[L] += 1
                nnf += 1

    def nrm(a, n):
        return (a / n).tolist() if n else []
    rr, cc = nrm(rec, ntar), nrm(ctl, ntar)
    net = [rr[L] - cc[L] for L in range(len(rr))]
    locus = next((L for L, v in enumerate(net) if v >= 0.3), None)
    result = {**art["result"], "n_flip_targets": ntar, "n_nonflip_controls": nnf,
              "recovery_rank1_by_layer": rr, "control_rank1_by_layer": cc,
              "net_rank1_by_layer": net, "nonflip_disruption_by_layer": nrm(disr, nnf),
              "locus_depth_net30": locus}
    print(f"[exp_e] flip targets={ntar} nonflip={nnf} locus={locus}")
    for L in range(depth):
        if rr:
            print(f"  L{L}: net {net[L]:+.2f} (rec {rr[L]:.2f} ctl {cc[L]:.2f})")
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(result, open(os.path.join(out, "experiment_e.json"), "w"), indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regime", default="adversarial")
    ap.add_argument("--max-clusters", type=int, default=60)
    ap.add_argument("--arch", default="gemma", choices=["gemma"])  # kept for run/ compatibility
    ap.add_argument("--grounding-token", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.steps, a.seed, a.regime, a.max_clusters, a.out,
        use_ground=a.grounding_token)


if __name__ == "__main__":
    main()
