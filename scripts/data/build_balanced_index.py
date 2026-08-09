"""Build a per-finding balanced PadChest index, so the question text predicts nothing.

The PadChest flip bank asks only about findings that are present. Every one of its 92
findings carries a single answer, so a lookup table on the finding name scores 861/861
and the benchmark cannot measure whether a model reads the radiograph. baby-MedGemma
confirms it: with all 256 vision tokens zeroed it still scores 1.000 there.

The raw question set has what the flip bank lacks. `padchest_questions.csv` lists, per
image, which findings are present. A negative is then any (image, finding) pair where
that finding is absent. Sampling an equal number of each per finding drives the
finding-name lookup ceiling to exactly 50%, which is the same construction
`build_transfer_index.py` uses and the reason the scaled model's blind accuracy is
0.503 rather than 1.000.

Paraphrases come from `babygemma.templates`, which excludes answer-inverting rewrites.

    python scripts/data/build_balanced_index.py \
        --questions-csv dataset/padchest/padchest_questions.csv \
        --images-dir cache/padchest_images \
        --out data/index_padchest_balanced.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random

from babygemma.templates import paraphrases_for


def split_for(uid: str, frac=(0.7, 0.15)) -> str:
    """Same hash split the rest of the pipeline uses, so splits stay comparable."""
    h = int(hashlib.md5(uid.encode()).hexdigest(), 16) % 1000 / 1000.0
    return "train" if h < frac[0] else ("val" if h < frac[0] + frac[1] else "test")


def build(questions_csv, images_dir, min_per_class=8, seed=0, require_image=True):
    rows = [r for r in csv.DictReader(open(questions_csv))
            if r.get("question_type") == "presence"]
    present = collections.defaultdict(set)
    for r in rows:
        fn, lab = r.get("filename"), (r.get("label") or "").strip().lower()
        if not fn or not lab:
            continue
        if require_image and not os.path.exists(os.path.join(images_dir, fn)):
            continue
        present[fn].add(lab)

    images = sorted(present)
    findings = collections.Counter(l for s in present.values() for l in s)
    rng = random.Random(seed)
    out, skipped = [], 0

    for finding in sorted(findings):
        pos = [i for i in images if finding in present[i]]
        neg = [i for i in images if finding not in present[i]]
        k = min(len(pos), len(neg))
        if k < min_per_class:
            skipped += 1
            continue
        rng.shuffle(pos)
        rng.shuffle(neg)
        paras = [{"text": p["text"], "phenomenon": p["phenomenon"],
                  "para_split": p["para_split"]} for p in paraphrases_for(finding)]
        for img, ans in [(i, "yes") for i in pos[:k]] + [(i, "no") for i in neg[:k]]:
            uid = f"pcbal:{finding}:{img}:{ans}"
            out.append({
                "uid": uid,
                "image_path": os.path.join(images_dir, img),
                "question": f"is there {finding}?",
                "answer": ans,
                "source": "padchest_balanced",
                "finding": finding,
                "paraphrases": paras,
                "split": split_for(uid),
            })
    return out, images, findings, skipped


def report(index):
    n = len(index)
    by = collections.defaultdict(list)
    for r in index:
        by[r["finding"]].append(1 if r["answer"] == "yes" else 0)
    mixed = sum(1 for v in by.values() if 0 < sum(v) < len(v))
    hit = sum(max(sum(v), len(v) - sum(v)) for v in by.values())
    print(f"[balanced] questions               : {n}")
    print(f"[balanced] findings                : {len(by)} ({mixed} carrying both answers)")
    print(f"[balanced] yes-rate                : {sum(sum(v) for v in by.values()) / n * 100:.1f}%")
    print(f"[balanced] finding lookup ceiling  : {hit / n * 100:.1f}%")
    print(f"[balanced] paraphrases per question: {len(index[0]['paraphrases'])}")
    for sp in ("train", "val", "test"):
        s = [r for r in index if r["split"] == sp]
        if s:
            print(f"[balanced]   {sp:5s} n={len(s):6d}  "
                  f"yes-rate={sum(r['answer'] == 'yes' for r in s) / len(s) * 100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-csv", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-per-class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-require-image", action="store_true",
                    help="keep questions whose image is not on disk yet")
    a = ap.parse_args()

    index, images, findings, skipped = build(
        a.questions_csv, a.images_dir, a.min_per_class, a.seed,
        require_image=not a.no_require_image)
    print(f"[balanced] images usable           : {len(images)}")
    print(f"[balanced] findings seen           : {len(findings)} "
          f"({skipped} skipped for fewer than {a.min_per_class} of one class)")
    report(index)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(index, fh)
    print(f"[balanced] wrote {a.out}")


if __name__ == "__main__":
    main()
