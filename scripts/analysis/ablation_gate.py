"""A gate for baby-MedGemma that can actually run in a clinic.

The published deployment rule needs two things a live case does not have: a second
image known to carry the opposite label, and a radiologist's box around the finding.
Both are offline checks. This script tests a replacement that needs neither.

baby-MedGemma reads 256 vision tokens that we can simply zero. So image reliance is
measurable per case with one extra forward pass:

    stability  = |margin|                      how far the answer sits from the boundary
    reliance   = |margin_seen - margin_blind|  how much of the answer came from the image

Admit only when both clear a threshold. The blind pass depends on the question tokens
alone, so its margin caches per distinct question string and costs nothing after warm-up.

We score the gate against the failure it is supposed to prevent: admitting a case the
model would have answered the same way with no image at all.

    NANO_INDEX=data/index_padchest_balanced.json \
    python scripts/analysis/ablation_gate.py --out results_audit/gate
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from babygemma import metrics as Mx
from babygemma.dataset import collate
from babygemma.training import train_model


@torch.no_grad()
def read_margins(model, dataset, device, batch_size=256, zero_vision=False):
    """Per row: the yes-minus-no margin, the prediction, and the bookkeeping."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    m, pred, ans, cl, ph = [], [], [], [], []
    for b in loader:
        vis = b["vision"].to(device)
        gnd = b["ground"].to(device) if "ground" in b else None
        if zero_vision:
            vis = torch.zeros_like(vis)
            if gnd is not None:
                gnd = torch.zeros_like(gnd)
        kw = {"ground": gnd} if getattr(model, "use_ground", False) else {}
        logits, _ = model(vis, b["tokens"].to(device), b["ans_pos"].to(device), **kw)
        m.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
        pred.extend(logits.argmax(-1).cpu().tolist())
        ans.extend(b["answer"].tolist())
        cl.extend(b["cluster_id"].tolist())
        ph.extend(b["phenomenon"])
    return (np.array(m), np.array(pred), np.array(ans), np.array(cl), ph)


def auc(score, pos):
    """Rank-based area under the curve. pos==1 marks what the score should rank high."""
    o = np.argsort(score, kind="mergesort")
    r = np.empty(len(score), float)
    s = score[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    np_, nn = pos.sum(), (~pos.astype(bool)).sum()
    if not np_ or not nn:
        return float("nan")
    return float((r[pos.astype(bool)].sum() - np_ * (np_ + 1) / 2) / (np_ * nn))


def evaluate(name, keep, correct, blind_same, blind_wrong, flip_row, n):
    """Rates among the admitted rows.

    `accuracy_on_grounded` is reported but is CIRCULAR and must not be quoted as
    evidence. With a binary answer, a row where the blind prediction differs from the
    sighted one and is wrong is a row where the sighted one is right, so this quantity
    is pinned to 1 - blind_same by arithmetic rather than by the gate working. The
    honest pair is `accuracy` and `image_dependent_share`: how often the gate is right,
    and how often the image actually changed the answer it admitted. On a per-finding
    balanced bank the text prior cannot answer anything, so `accuracy` already is
    accuracy on grounded cases.
    """
    k = int(keep.sum())
    if not k:
        return {"rule": name, "admitted": 0, "coverage": 0.0}
    return {
        "rule": name,
        "admitted": k,
        "coverage": k / n,
        "accuracy": float(correct[keep].mean()),
        "image_dependent_share": float(1.0 - blind_same[keep].mean()),
        "blind_would_answer_the_same": float(blind_same[keep].mean()),
        "grounded_share": float(blind_wrong[keep].mean()),
        "accuracy_on_grounded_CIRCULAR": (float(correct[keep & blind_wrong].mean())
                                          if (keep & blind_wrong).sum() else float("nan")),
        "n_grounded": int((keep & blind_wrong).sum()),
        "flip_rate_admitted": float(flip_row[keep].mean()),
    }


def run(steps, seed, regime, out, use_ground=False):
    art = train_model(regime=regime, seed=seed, steps=steps, use_ground=use_ground)
    model, ds, device = art["model"], art["eval_ds"], art["device"]

    m_seen, p_seen, ans, cl, ph = read_margins(model, ds, device)
    m_blind, p_blind, _, _, _ = read_margins(model, ds, device, zero_vision=True)

    correct = (p_seen == ans)
    blind_same = (p_blind == p_seen)
    blind_wrong = (p_blind != ans)          # the text prior alone fails: a grounded case
    stability = np.abs(m_seen)
    reliance = np.abs(m_seen - m_blind)

    flips = Mx.flip_labels(list(p_seen), list(cl))
    flip_row = np.array([bool(flips[c]) for c in cl])
    n = len(ans)

    print(f"[gate] {n} eval rows, {len(set(cl.tolist()))} clusters")
    print(f"[gate] sighted accuracy {correct.mean():.3f}   "
          f"blind accuracy {(p_blind == ans).mean():.3f}")
    print(f"[gate] blind gives the same answer on {blind_same.mean() * 100:.1f}% of rows")
    print(f"[gate] grounded rows (text prior alone is wrong): {blind_wrong.mean() * 100:.1f}%")

    aucs = {
        "stability_for_flips": auc(-stability, flip_row.astype(int)),
        "stability_for_grounded": auc(stability, blind_wrong.astype(int)),
        "reliance_for_grounded": auc(reliance, blind_wrong.astype(int)),
        "reliance_for_flips": auc(-reliance, flip_row.astype(int)),
    }
    print(f"[gate] |margin| ranks flips at        {aucs['stability_for_flips']:.3f}")
    print(f"[gate] |margin| ranks grounded at     {aucs['stability_for_grounded']:.3f}"
          f"   <- the blind spot")
    print(f"[gate] ablation delta ranks grounded  {aucs['reliance_for_grounded']:.3f}"
          f"   <- the replacement signal")

    # thresholds chosen on the validation half of the clusters, applied to the rest
    cls = sorted(set(cl.tolist()))
    rng = np.random.default_rng(0)
    pick = set(rng.permutation(cls)[: len(cls) // 2].tolist())
    tune = np.array([c in pick for c in cl])
    hold = ~tune

    def q(x, p):
        return float(np.quantile(x[tune], p))

    rules = []
    for cov in (0.5, 0.33, 0.2):
        tau = q(stability, 1 - cov)
        rules.append((f"margin only, {int(cov * 100)}% coverage", stability >= tau))
        d = q(reliance, 1 - cov)
        rules.append((f"ablation delta only, {int(cov * 100)}% coverage", reliance >= d))
        tau2 = q(stability, 1 - cov ** 0.5)
        d2 = q(reliance, 1 - cov ** 0.5)
        rules.append((f"both, {int(cov * 100)}% coverage",
                      (stability >= tau2) & (reliance >= d2)))
    rules.append(("no gate", np.ones(n, bool)))
    rules.append(("cluster is stable under paraphrase", ~flip_row))

    report = []
    for name, keep in rules:
        k = keep & hold
        report.append(evaluate(name, k, correct, blind_same, blind_wrong, flip_row,
                               int(hold.sum())))

    base = float(correct[hold].mean())
    print(f"\n[gate] held-out half, {int(hold.sum())} rows. "
          f"Answering everything scores {base * 100:.1f}%.")
    print(f"  {'rule':40s} {'admit':>7s} {'acc':>7s} {'vs base':>8s} {'image-dependent':>16s}")
    for r in report:
        if not r["admitted"]:
            print(f"  {r['rule']:40s}  admits nothing")
            continue
        print(f"  {r['rule']:40s} {r['coverage'] * 100:6.1f}% {r['accuracy'] * 100:6.1f}% "
              f"{(r['accuracy'] - base) * 100:+7.1f} {r['image_dependent_share'] * 100:15.1f}%")

    distinct_q = len({(ph[i], int(cl[i])) for i in range(n)})
    result = {**art["result"], "n_eval_rows": n, "auc": aucs, "rules": report,
              "accuracy_seen": float(correct.mean()),
              "accuracy_blind": float((p_blind == ans).mean()),
              "blind_same_rate": float(blind_same.mean()),
              "grounded_rate": float(blind_wrong.mean()),
              "distinct_question_strings_for_blind_cache": distinct_q}
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "ablation_gate.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n[gate] saved {out}/ablation_gate.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regime", default="augmented")
    ap.add_argument("--grounding-token", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.steps, a.seed, a.regime, a.out, use_ground=a.grounding_token)


if __name__ == "__main__":
    main()
