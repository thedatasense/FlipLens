"""How much of this benchmark can be answered without looking at the image?

Trains baby-MedGemma on whatever index NANO_INDEX points at, then scores the same
evaluation clusters twice: normally, and with the vision tokens zeroed. Three
quantities come out of that pair, and they answer three different questions.

  blind accuracy        can the text prior alone get the answer right?
  blind/sighted agree   does the image change the answer at all?
  lookup ceiling        does the finding name already determine the answer?

The third needs no model. If every question about a finding carries the same label,
a lookup table on the finding name scores 100% and the benchmark cannot measure
grounding, whatever the model does.

    NANO_INDEX=data/index_padchest.json \
    python scripts/analysis/text_only_audit.py --out results_audit/padchest
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

from babygemma import metrics as Mx
from babygemma.data_index import build_index
from babygemma.training import train_model


def lookup_ceiling(index, split):
    """What a table keyed on the finding name alone scores on this split."""
    rows = [r for r in index if r["split"] == split]
    key = lambda r: r.get("finding") or r.get("question")
    by = collections.defaultdict(list)
    for r in rows:
        by[key(r)].append(1 if r["answer"] == "yes" else 0)
    hit = sum(max(sum(v), len(v) - sum(v)) for v in by.values())
    mixed = {k: v for k, v in by.items() if 0 < sum(v) < len(v)}
    yes = sum(sum(v) for v in by.values())
    return {
        "n": len(rows),
        "n_findings": len(by),
        "n_findings_with_both_answers": len(mixed),
        "n_questions_in_mixed_findings": sum(len(v) for v in mixed.values()),
        "finding_lookup_accuracy": hit / len(rows) if rows else 0.0,
        "majority_class_accuracy": max(yes, len(rows) - yes) / len(rows) if rows else 0.0,
        "yes_rate": yes / len(rows) if rows else 0.0,
    }


def run(steps, seed, regime, out, use_ground=False):
    index = build_index()
    split = os.environ.get("NANO_EVAL_SPLIT", "test")
    ceiling = lookup_ceiling(index, split)
    print(f"[audit] {ceiling['n']} eval questions, {ceiling['n_findings']} findings, "
          f"{ceiling['n_findings_with_both_answers']} of them carrying both answers")
    print(f"[audit] finding-name lookup ceiling: {ceiling['finding_lookup_accuracy']:.3f}")

    art = train_model(regime=regime, seed=seed, steps=steps, use_ground=use_ground)
    model, ds, device = art["model"], art["eval_ds"], art["device"]

    seen, ans, clusters, phen = Mx.predict(model, ds, device)
    blind, _, _, _ = Mx.predict(model, ds, device, zero_vision=True)
    seen, blind, ans = np.array(seen), np.array(blind), np.array(ans)

    # Per register, how often does a phrasing disagree with its cluster's original?
    # negation_pattern is the one to read: those rewrites inverted the answer, so
    # agreeing with the original means the model did not notice the negation.
    orig = {}
    for i, c in enumerate(clusters):
        if phen[i] == "original":
            orig[c] = seen[i]
    per_phen = collections.defaultdict(lambda: [0, 0])
    for i, c in enumerate(clusters):
        if phen[i] == "original" or c not in orig:
            continue
        per_phen[phen[i]][1] += 1
        if seen[i] != orig[c]:
            per_phen[phen[i]][0] += 1
    disagree = {k: v[0] / v[1] for k, v in per_phen.items() if v[1]}
    counts = {k: v[1] for k, v in per_phen.items()}

    agree = float((seen == blind).mean())
    flips_seen = Mx.flip_labels(list(seen), clusters)
    flips_blind = Mx.flip_labels(list(blind), clusters)

    # the deployment-gate analogue: among clusters that answer consistently under
    # rephrasing, how many does the blind model answer the same way?
    by = collections.defaultdict(list)
    for i, c in enumerate(clusters):
        by[c].append(i)
    stable = [c for c, f in flips_seen.items() if not f]
    stable_agree = float(np.mean([
        all(seen[i] == blind[i] for i in by[c]) for c in stable
    ])) if stable else 0.0

    result = {
        **art["result"],
        "eval_split": split,
        "lookup": ceiling,
        "accuracy_seen": float((seen == ans).mean()),
        "accuracy_blind": float((blind == ans).mean()),
        "grounding_gap": float((seen == ans).mean() - (blind == ans).mean()),
        "blind_agrees_with_seen": agree,
        "flip_rate_seen": Mx.flip_rate(list(seen), clusters),
        "flip_rate_blind": Mx.flip_rate(list(blind), clusters),
        "n_stable_clusters": len(stable),
        "stable_clusters_blind_agrees": stable_agree,
        "n_eval_rows": int(len(seen)),
        "disagreement_with_original_by_register": disagree,
        "rows_by_register": counts,
    }

    print(f"[audit] sighted accuracy   {result['accuracy_seen']:.3f}")
    print(f"[audit] blind accuracy     {result['accuracy_blind']:.3f}   "
          f"(gap {result['grounding_gap']:+.3f})")
    print(f"[audit] blind agrees with sighted on {agree * 100:.1f}% of rows")
    print(f"[audit] among {len(stable)} stable clusters, blind agrees on "
          f"{stable_agree * 100:.1f}%")
    print(f"[audit] flip rate  sighted {result['flip_rate_seen']:.3f}  "
          f"blind {result['flip_rate_blind']:.3f}")
    for k in sorted(disagree, key=lambda x: -disagree[x]):
        note = "  <- these inverted the answer, so a low rate means the model missed it" \
            if k == "negation_pattern" else ""
        print(f"[audit]   {k:24s} disagrees with the original on "
              f"{disagree[k] * 100:5.1f}% of {counts[k]} rows{note}")

    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "text_only_audit.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[audit] saved {out}/text_only_audit.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regime", default="augmented")
    ap.add_argument("--grounding-token", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.steps, a.seed, a.regime, a.out, use_ground=a.grounding_token)


if __name__ == "__main__":
    main()
