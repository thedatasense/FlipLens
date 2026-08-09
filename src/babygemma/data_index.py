"""Unified index of real chest X-ray binary VQA with register-tagged paraphrases.

Sources (both already on disk, both carry a `phenomenon` register tag per
paraphrase, so the three training regimes can be built by re-bucketing):
  - MIMIC-CXR presence questions: dataset/mimic/splits/{train,val,test}.json
  - PadChest flip bank:            dataset/padchest/padchest_flip_bank.csv

Each unified record:
  {uid, image_path, question, answer('yes'/'no'), source,
   paraphrases: [{text, phenomenon}], split}

Only records whose image resolves on disk are kept. This is the substrate for
the MedSigLIP nanoVLM PSF experiments (docs/tiny_vlm_psf_isolation.md, rung 2).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict

PROJECT = "/home/bsada1/medical-vlm-robustness"
MIMIC_IMG = os.environ.get("MIMIC_IMAGE_DIR", "/home/bsada1/datasets/MIMIC_JPG/thousandfiles")
PHENOMENA = ["lexical_substitution", "syntactic_restructuring",
             "negation_pattern", "scope_quantification", "specificity_modulation"]


def _split_for(uid: str, frac=(0.7, 0.15)) -> str:
    h = int(hashlib.md5(uid.encode()).hexdigest(), 16) % 1000 / 1000.0
    return "train" if h < frac[0] else ("val" if h < frac[0] + frac[1] else "test")


def load_mimic() -> list[dict]:
    out = []
    for split in ["train", "val", "test"]:
        p = os.path.join(PROJECT, f"dataset/mimic/splits/{split}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rows = d["results"] if isinstance(d, dict) and "results" in d else d
        for r in rows:
            if r.get("question_type") != "presence":
                continue
            ans = str(r.get("answer", "")).lower()
            if ans not in ("yes", "no"):
                continue
            fn = r.get("filename")
            img = os.path.join(MIMIC_IMG, fn) if fn else None
            if not img or not os.path.exists(img):
                continue
            paras = []
            for pp in r.get("paraphrases", []):
                if isinstance(pp, dict) and pp.get("text"):
                    paras.append({"text": pp["text"], "phenomenon": pp.get("phenomenon")})
            out.append({
                "uid": f"mimic:{r.get('question_id')}",
                "image_path": img,
                "question": r.get("original_question"),
                "answer": ans,
                "source": "mimic",
                "paraphrases": paras,
                "split": split,
            })
    return out


def load_padchest() -> list[dict]:
    """PadChest flip bank -> one record per question, with its paraphrase cluster.

    Rows carrying `op_match == "False"` are dropped. Those are the 750
    `negation_pattern` rewrites that turn a presence question into an exclusion one
    ("Can you rule out aortic elongation?" against "Is there aortic elongation?"),
    so they INVERT the answer and cannot share a cluster with the original. 87.1% of
    the 861 clusters contain one. This is the mixed-question defect that
    `templates.py` documents and that the newer template bank already avoids.

    Set NANO_KEEP_OPERATOR_VARIANTS=1 to restore the previous behaviour, which is
    what the retired 1,841-question probe under results_gemma/ was built with.
    """
    p = os.path.join(PROJECT, "dataset/padchest/padchest_flip_bank.csv")
    if not os.path.exists(p):
        return []
    keep_inverting = bool(os.environ.get("NANO_KEEP_OPERATOR_VARIANTS"))
    rows = list(csv.DictReader(open(p)))
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["question_id"]].append(r)
    out = []
    for qid, rs in by_q.items():
        r0 = rs[0]
        img = r0.get("image_path")
        if not img or not os.path.exists(img):
            continue
        ans = str(r0.get("ground_truth", "")).lower()
        if ans not in ("yes", "no"):
            continue
        paras = [{"text": r["paraphrase"], "phenomenon": r.get("phenomenon")}
                 for r in rs if r.get("paraphrase")
                 and (keep_inverting or str(r.get("op_match", "True")) != "False")]
        uid = f"padchest:{qid}"
        out.append({
            "uid": uid,
            "image_path": img,
            "question": r0.get("original_question"),
            "answer": ans,
            "source": "padchest",
            "paraphrases": paras,
            "split": _split_for(uid),
        })
    return out


def build_index() -> list[dict]:
    # explicit index wins (NANO_INDEX=data/index_scaled.json for the scaled run),
    # so the 1,841-question experiments stay reproducible untouched
    override = os.environ.get("NANO_INDEX")
    if override:
        idx = json.load(open(override))
        return [r for r in idx if r.get("question")]
    # prefer the expanded 10k index (PadChest label-derived + MIMIC) if built
    from babygemma.paths import DATA
    expanded = os.path.join(DATA, "index_10k.json")
    if os.path.exists(expanded):
        idx = json.load(open(expanded))
    else:
        idx = load_mimic() + load_padchest()
    idx = [r for r in idx if r.get("question")]
    return idx


def unique_images(index: list[dict]) -> list[str]:
    return sorted({r["image_path"] for r in index})


if __name__ == "__main__":
    idx = build_index()
    from collections import Counter
    print("total records:", len(idx))
    print("by source:", dict(Counter(r["source"] for r in idx)))
    print("by split:", dict(Counter(r["split"] for r in idx)))
    print("by answer:", dict(Counter(r["answer"] for r in idx)))
    print("unique images:", len(unique_images(idx)))
    print("avg paraphrases:", sum(len(r["paraphrases"]) for r in idx) / max(1, len(idx)))
    print("sample:", {k: (v if k != "paraphrases" else v[:1]) for k, v in idx[0].items()})
