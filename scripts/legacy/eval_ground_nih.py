"""Does giving the decoder a real grounding token fix the near-chance NIH result?

Trains baby-Gemma with and without the MedSigLIP attention-pooled grounding token
and evaluates both on the same balanced NIH set: accuracy, flip rate, and the
per-finding answer distribution (the tell for text-prior collapse).

    CUDA_VISIBLE_DEVICES=0 python eval_ground_nih.py
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import torch

from babygemma.paths import ROOT as HERE

from babygemma.dataset import Ex, NanoDataset, build_tokenizer
from babygemma import metrics as Mx
from babygemma import nih as ND
from babygemma.encoders import encode_pooled
from babygemma.training import train_model

SEED, PER = 0, 120
NIH_FEATS = os.path.join(HERE, "cache", "nih_feats.pt")
NIH_POOLED = os.path.join(HERE, "cache", "nih_pooled.pt")
OUT = os.path.join(HERE, "results_gemma", "grounding", "ground_nih.json")


def nih_data(tok):
    records, _ = ND.build_nih_records(tok, PER, SEED)
    paths = sorted({r["image_path"] for r in records})
    feats = torch.load(NIH_FEATS, map_location="cpu") if os.path.exists(NIH_FEATS) \
        else ND.precompute_nih(paths)
    if os.path.exists(NIH_POOLED):
        d = torch.load(NIH_POOLED, map_location="cpu")
    else:
        print(f"[nih] encoding pooled embeddings for {len(paths)} images...", flush=True)
        kept, pooled = encode_pooled(paths)
        d = {"paths": kept, "pooled": pooled}
        torch.save(d, NIH_POOLED)
    pc = {"index": {p: i for i, p in enumerate(d["paths"])}, "pooled": d["pooled"]}
    return records, feats, pc


def evaluate(model, records, feats, pc, tok, use_ground):
    """Full paraphrase clusters -> accuracy, flip rate; originals -> per-finding yes-rate."""
    ex, orig_mask = [], []
    for c, r in enumerate(records):
        ex.append(Ex(r["image_path"], r["question"], r["answer"], "original", c)); orig_mask.append(True)
        for p in r["paraphrases"]:
            ex.append(Ex(r["image_path"], p["text"], r["answer"], p.get("phenomenon"), c))
            orig_mask.append(False)
    ds = NanoDataset(ex, tok, feats, pooled=pc if use_ground else None)
    preds, ans, clusters, _ = Mx.predict(model, ds, "cuda")
    preds = np.array(preds); ans = np.array(ans); om = np.array(orig_mask)
    per_finding = {}
    q = [records[c]["question"] for c in clusters]
    for question in sorted(set(q)):
        m = np.array([x == question for x in q]) & om
        if m.sum():
            per_finding[question] = {"acc": float((preds[m] == ans[m]).mean()),
                                     "frac_yes": float(preds[m].mean()), "n": int(m.sum())}
    return {
        "acc_all": float((preds == ans).mean()),
        "acc_originals": float((preds[om] == ans[om]).mean()),
        "frac_yes_originals": float(preds[om].mean()),
        "flip_rate": Mx.flip_rate(list(preds), clusters),
        "per_finding": per_finding,
    }


def main():
    tok = build_tokenizer()
    records, feats, pc = nih_data(tok)
    print(f"[nih] {len(records)} clusters", flush=True)

    out = {}
    for use_ground in (False, True):
        tag = "grounded" if use_ground else "baseline"
        print(f"\n=== training {tag} (augmented, seed {SEED}) ===", flush=True)
        art = train_model(regime="augmented", seed=SEED, steps=2000, lr=5e-4,
                          use_ground=use_ground, verbose=False)
        native = art["result"]
        res = evaluate(art["model"], records, feats, pc, tok, use_ground)
        out[tag] = {"native": {k: native[k] for k in
                               ("accuracy", "accuracy_vision_ablated", "grounding_gap", "flip_rate")},
                    "nih": res}
        n, r = out[tag]["native"], res
        print(f"[{tag}] NATIVE acc={n['accuracy']:.3f} gap={n['grounding_gap']:.3f} flip={n['flip_rate']:.3f}")
        print(f"[{tag}] NIH    acc={r['acc_originals']:.3f} flip={r['flip_rate']:.3f} "
              f"frac_yes={r['frac_yes_originals']:.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    print("\nper-finding NIH accuracy (baseline -> grounded):")
    for q in sorted(out["baseline"]["nih"]["per_finding"]):
        b = out["baseline"]["nih"]["per_finding"][q]; g = out["grounded"]["nih"]["per_finding"][q]
        print(f"  {q:28s} {b['acc']:.2f} (yes {b['frac_yes']:.2f})  ->  "
              f"{g['acc']:.2f} (yes {g['frac_yes']:.2f})")


if __name__ == "__main__":
    main()
