"""Experiment A (nanoVLM): divergence trajectory + register-axis attribution.

Per-layer within-cluster dispersion vs flip status, plus which register
(phenomenon) carries the flips. Localizes where PSF originates vs is amplified,
on real chest X-rays with the frozen MedSigLIP vision encoder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader

from babygemma.paths import ROOT as HERE

from babygemma import metrics as Mx
from babygemma.dataset import collate
from babygemma.training import train_model


@torch.no_grad()
def predict_with_phen(model, dataset, device, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    rows = []
    for b in loader:
        logits, _ = model(b["vision"].to(device), b["tokens"].to(device), b["ans_pos"].to(device))
        preds = logits.argmax(-1).cpu().tolist()
        for j, p in enumerate(preds):
            rows.append({"cluster": int(b["cluster_id"][j]), "pred": p, "phen": b["phenomenon"][j]})
    return rows


def phen_disagreement(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["cluster"]].append(r)
    dis = defaultdict(lambda: [0, 0])
    for rs in by.values():
        maj = Counter(r["pred"] for r in rs).most_common(1)[0][0]
        for r in rs:
            dis[r["phen"]][1] += 1
            if r["pred"] != maj:
                dis[r["phen"]][0] += 1
    return {k: (d / t if t else 0.0) for k, (d, t) in dis.items()}


def run(steps, seed, regime, out):
    art = train_model(regime=regime, seed=seed, steps=steps)
    model, ds, device = art["model"], art["eval_ds"], art["device"]
    preds, _, clusters, _ = Mx.predict(model, ds, device)
    flips = Mx.flip_labels(preds, clusters)
    disp = Mx.layerwise_dispersion(model, ds, device)
    per_layer = {li: Mx.point_biserial(disp[li], flips) for li in sorted(disp)}
    rows = predict_with_phen(model, ds, device)
    result = {**art["result"],
              "per_layer_dispersion_flip_corr": per_layer,
              "flip_fraction": sum(flips.values()) / len(flips) if flips else 0.0,
              "phenomenon_disagreement": phen_disagreement(rows)}
    print("[exp_a] per-layer dispersion-vs-flip corr:",
          {li: round(c, 2) for li, c in per_layer.items()})
    print("[exp_a] phenomenon disagreement:",
          {k: round(v, 3) for k, v in result["phenomenon_disagreement"].items()})
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(result, open(os.path.join(out, "experiment_a.json"), "w"), indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regime", default="augmented")
    ap.add_argument("--arch", default="gemma", choices=["gemma"])  # kept for run/ compatibility
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.steps, a.seed, a.regime, a.out)


if __name__ == "__main__":
    main()
