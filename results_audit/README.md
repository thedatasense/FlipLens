# Benchmark validity, the vision-ablation gate, and lens faithfulness

Results from a set of experiments run on baby-MedGemma in August 2026. Everything here
**adds** numbers. No published result was recomputed and no published figure was
regenerated.

Three things came out of it:

| # | Finding | Where it lands |
|---|---|---|
| 1 | The PadChest flip bank cannot measure image grounding. Its 861 questions span 92 findings and no finding carries both answers, so a lookup table on the finding name scores 861/861. baby-MedGemma scores 1.000 on it with every vision token zeroed. | Chapter 7, and it reframes the gate result |
| 2 | A per-finding balanced rebuild fixes it. On the rebuilt bank the model's blind accuracy falls to 0.497 and it earns +0.120 from the image. A gate on the vision-ablation delta then admits cases that are 99.7% image-dependent. | Chapter 7 |
| 3 | The Jacobian lens is faithful and it predicts the patch it corroborates. Lens against true margin reaches r = 0.995, and the attribution-patching regression reaches R² = 0.998. | Chapter 5 |

---

## 1. The benchmark cannot measure grounding

Read straight off the labels in `dataset/padchest/padchest_flip_bank.csv`. No model
needed.

| Bank | Binary questions | Findings | Findings carrying both answers | Finding-name lookup ceiling |
|---|---|---|---|---|
| PadChest flip bank | 861 | 92 | **0** | **100.0%** |
| VinDr flip bank | 25,935 | 14 | **0** | **100.0%** (all yes) |
| Rebuilt balanced bank | 17,144 | 97 | **97** | **50.0%** |

Ninety-one PadChest findings are always yes and `Normal` is always no. So the question
text alone determines the answer, and any gate that selects confident, stable
predictions is selecting the text prior. That's provable before a model runs.

VinDr is worse. Every one of its 25,935 binary questions is a yes.

Three independent measurements agree:

| Measurement | Value |
|---|---|
| Finding-name lookup ceiling on the bank | 861/861 = 100.0% |
| MedGemma-4B text-only accuracy, recovered from the published gate rows | 98.0% base, 98.4% targeted LoRA |
| baby-MedGemma blind accuracy on the same bank | **1.000** |

The third is the one to quote at a defense. A 13.7M-parameter decoder with all 256
vision tokens zeroed reaches 1.000 in 75 training steps.

### What this does to the Chapter 7 claim

The current claim is that a gate cannot admit image-grounded cases. That invites the
reply "your benchmark had none," and the reply would be right. The version that holds:

> On a bank where the finding name determines the answer, a multi-signal gate admits
> the text-answerable cases and nothing else, because there is nothing else to admit.
> That is a property of the benchmark, provable from the labels. On a per-finding
> balanced bank the same question has a different answer.

That turns a negative result into a benchmark-validity result, which is a contribution
rather than a shortfall.

### An answer-inverting contamination in the same file

750 of the 4,251 paraphrase rows carry `op_match = False`. All are `negation_pattern`,
all convert a presence question into an exclusion one ("Can you rule out aortic
elongation?" against "Is there aortic elongation?"), so they invert the answer and
cannot share a paraphrase cluster with the original. They sit in **87.1% of the 861
clusters**. `templates.py` already documents this as the mixed-question defect and the
newer template bank avoids it; the PadChest loader did not.

It suppresses flip rate rather than inflating it. baby-MedGemma disagrees with the
original on **0.0% of the 132 answer-inverting evaluation rows**, so it never notices
the negation, and the contaminated index reports flip 0.000 against 0.015 for the
filtered one. Negation-blindness reads as stability.

`load_padchest` now filters those rows. `NANO_KEEP_OPERATOR_VARIANTS=1` restores the
old behaviour, so `results_gemma/` stays reproducible. The 107 negation rows with
`op_match = True` are double negations ("evidence against the absence of X") that
preserve the answer, and they're kept.

---

## 2. The rebuilt bank, and what the model does on it

`scripts/data/build_balanced_index.py` reads `padchest_questions.csv`, which lists
which findings are present on each image. A negative is any (image, finding) pair where
that finding is absent. Sampling equal counts per finding drives the ceiling to 50%.
This is the same construction `build_transfer_index.py` already applies, which is why
the scaled model's blind accuracy is 0.503 rather than 1.000.

```
images usable          4555
questions             17144
findings                 97   (97 carrying both answers)
yes-rate               50.0%
lookup ceiling         50.0%
train 12004 / val 2579 / test 2561
```

baby-MedGemma on each bank, same architecture, same recipe:

| | Flip bank | Balanced bank |
|---|---|---|
| Lookup ceiling | 100.0% | 56.0% on the test split |
| Sighted accuracy | 1.000 | 0.617 |
| **Blind accuracy** | **1.000** | **0.497** |
| Grounding gap | 0.000 | **+0.120** |
| Blind agrees with sighted | 100.0% | 48.6% |
| Flip rate, sighted | 0.000 | 0.176 |
| Flip rate, blind | 0.000 | **0.000** |

Blind accuracy of 0.497 is chance. The model now has to read the radiograph and earns
+0.120 by doing it.

Two details worth carrying into the text. The test-split ceiling is 56.0% rather than
50.0% because the bank balances per finding globally while the hash split doesn't
preserve that inside each finding; 7 of 94 test findings came out pure. And flip rate
is 0.176 sighted against **0.000 blind**: with the image zeroed the model answers every
paraphrase of a question identically. Paraphrase sensitivity only appears once the image
is in play, so flips are not purely a language-side artifact on this bank.

---

## 3. A gate that a clinic could run

The published rule needs a second image known to carry the opposite label and a
radiologist's box around the finding. Both are offline checks. baby-MedGemma doesn't
need either, because its 256 vision tokens can simply be zeroed:

```
stability = |margin|                        distance from the decision boundary
reliance  = |margin_seen - margin_blind|    how much of the answer came from the image
```

The blind pass depends only on the question tokens, so its margin caches per distinct
question string and costs nothing after warm-up.

Ranking power, 125,489 evaluation rows:

| Signal | Ranks paraphrase flips | Ranks image dependence |
|---|---|---|
| `\|margin\|` | **0.878** | **0.500** |
| Ablation delta | 0.498 | **0.617** |

The margin sits at 0.500 for image dependence. That reproduces the dissertation's
0.47-to-0.52 blind spot on a second architecture and a rebuilt benchmark.

Held-out half, 62,769 rows. Answering everything scores 60.6%:

| Rule | Admitted | Accuracy | vs base | Image changed the answer |
|---|---|---|---|---|
| No gate | 100% | 60.6% | baseline | 47.1% |
| Paraphrases agree | 82.8% | 62.6% | +2.0 | 46.7% |
| Margin only, 20% coverage | 20.6% | **72.5%** | **+11.8** | 57.2% |
| Ablation delta, 20% coverage | 20.9% | 67.8% | +7.2 | **99.7%** |
| Both, 20% coverage | 22.9% | 68.0% | +7.4 | 93.6% |

The two signals do different jobs, and that's the result. The margin buys accuracy. The
delta buys grounding: at 20% coverage it admits cases where the image changed the answer
99.7% of the time, against 47.1% with no gate, while still beating the baseline by 7.2
points.

So the answer to "can a gate admit image-grounded cases?" has two halves. No on a bank
with no grounded cases. Yes on a balanced one, cheaply, per case.

### A metric to avoid

An earlier version of this analysis reported "accuracy on grounded cases" of 97.4%.
That number is circular and must not be quoted. With a binary answer, a row where the
blind prediction differs from the sighted one and is wrong is a row where the sighted one
is right, so the quantity is pinned to `1 - blind_same` by arithmetic. It survives in the
JSON as `accuracy_on_grounded_CIRCULAR`. The honest pair is accuracy on admitted and the
image-dependent share.

---

## 4. The Jacobian lens is faithful, and it predicts the patch

`jlens.py` never checked that its readout tracks the model, and `experiment_e.py` runs
the patch the lens approximates while keeping only the argmax. Both gaps are now closed
by `scripts/experiments/jlens_faithfulness.py`, on 80 donor-target pairs.

| Layer | Lens vs true margin (r) | Patch validation R² | Slope |
|---|---|---|---|
| 0 | 0.575 | 0.594 | 1.274 |
| 1 | 0.970 | 0.990 | 1.076 |
| 2 | 0.981 | 0.998 | 1.080 |
| 3 | 0.992 | 0.998 | 1.084 |
| 4 | 0.995 | 0.997 | 1.078 |
| 5 | 0.995 | 0.998 | **4.288** |

From layer 1 onward the lens reading correlates with the true yes-minus-no margin at
r ≥ 0.97, and its donor-target difference predicts the rank-1 patch's measured margin
change at R² ≥ 0.99 with slope near 1.08.

The published reference point is Syed et al. (arXiv:2310.10348) at R² = 0.27 and slope
0.531 for attribution patching against activation patching. Don't oversell the gap: that
study measured fine-grained edges in a much deeper model against a task metric, while
this is a 6-layer decoder with a linear two-way readout. The defensible statement is that
on this model the lens is a near-exact predictor of the intervention it approximates, and
that it degrades exactly where theory says it should, at layer 0 where the most
nonlinearity sits between the reading and the readout.

**The layer-5 slope of 4.288 is a normalization artifact, not a mechanism.** R² stays at
0.998, so the relationship is still linear; only the scale is wrong. The margin is
degree-zero homogeneous in the pre-norm layer-5 output, verified against
`transformers 4.57.1`: scaling that vector by 100 leaves the margin bit-identical, and
the gradient is orthogonal to the activation to 1.9e-09.

The same root cause explains an anomaly already in the committed results. At the last
layer the rank-1 patch should be deterministic and recovery should be exactly 1.000,
because only the final norm sits between the injected vector and the logits.
`experiment_e.py` harvests its donor from `acts[L]`, and `out.hidden_states[1:]` gives
`(out_L0 … out_L4, norm(out_L5))`, so the last entry is post-norm while the patch hook
writes pre-norm. Across all 34 committed runs the split is exact:

| Architecture | Runs | Layer-5 recovery |
|---|---|---|
| retired nanoVLM (`results/E/`) | 16 | **1.000 in every run** |
| baby-MedGemma (`results_gemma/E/`, `results_transfer_patch/`) | 18 | **0.273 to 0.817** |

The dip appears only where the last captured state is post-norm. Layers 0 to 4 are
unaffected: `acts[L]` was verified identical to the hooked layer output there. The fix is
one line, and it belongs after the defense.

---

## 5. Seeds, and a statistic that doesn't survive them

`jlens` and the SAE had each been run once, at seed 0, against 8 to 24 seeds elsewhere in
the project. Eight seeds on the balanced bank:

```
accuracy       0.610 ± 0.009
flip rate      0.162 ± 0.026
grounding gap  0.109 ± 0.013
```

| Layer | Flip ÷ stable ratio | Point-biserial with flip |
|---|---|---|
| 0 | 1.97 ± 0.50 | 0.290 ± 0.094 |
| 1 | 2.07 ± 0.42 | 0.329 ± 0.071 |
| 2 | 2.11 ± 0.41 | 0.357 ± 0.055 |
| 3 | 2.10 ± 0.37 | **0.368 ± 0.048** |
| 4 | 2.04 ± 0.38 | 0.367 ± 0.055 |
| 5 | 1.95 ± 0.38 | 0.357 ± 0.062 |

**The point-biserial rises with depth here**, 0.290 to 0.368, where the published probe
falls (0.809 to 0.769) and so does the scaled model (0.265 to 0.231). On a bank where the
answer needs the image, the flip signal accumulates through the layers instead of being
fully present at layer 0.

**`lens_commit_layer` is `[1, None, None, None, None, 0, 0, 0]` across the eight seeds.**
Five of eight return nothing. The rule fires when the ratio clears 2, and the ratio is
2.0 ± 0.4, so half the seeds fall under the threshold. Reporting that statistic from a
single seed was luck. Retire it as a headline and report the ratio curve with its
interval, which says the same thing without depending on a hardcoded constant.

### The sparse autoencoder on the balanced bank

Four seeds so far, and the picture is different from the published one:

| Quantity | Balanced bank (4 seeds) |
|---|---|
| SAE best feature, \|cos\| with the flip direction | 0.730 ± 0.041 |
| PCA top-20, \|cos\| | **0.773 ± 0.180** |
| Random direction, \|cos\| | 0.039 ± 0.002 |
| Best flip point-biserial | **0.016 ± 0.191** |
| Variance explained | 0.993 ± 0.003 |

Two things to state plainly. PCA now edges out the SAE, and the feature that best
predicts flips has a point-biserial of 0.016 with a standard deviation of 0.191, which is
indistinguishable from zero. On this bank the unsupervised decomposition still recovers
the causal flip axis well above a random direction, but it does **not** independently
predict flips. That is weaker than the published probe result and should be reported as
such.

---

## 6. What contradicts an existing claim

Three things, all small, all better said by you than found by a committee.

| Claim as written | What the data shows | Fix |
|---|---|---|
| The patch tokens alone leave a decoder this small at chance, so the grounding token supplies the finding signal | True at 795 images. At 4,552 images with balancing the model reaches 0.615 sighted with **no** grounding token against 0.602 with it, inside a ±0.009 seed spread | Scope the claim to the setting it was measured in |
| `figures/jlens_concept.png` shows two phrasings tracking together then splitting | The curves are literals in `figures.py`, and the measurement beside it reports 8.9× separation at layer 0 with a flat profile | Caption it a schematic. Already done in the README |
| The flip bank measures paraphrase robustness | It measures it on a bank whose answers are fully determined by the finding name | Report the lookup ceiling alongside the flip rate |

The `saillab/babymedgemma` repository currently holds only a 56 MB `model.safetensors`,
so the README's claim that the ~3 GB feature cache and the ~4 GB checkpoints live there
is not true right now.

---

## 7. Files

| Path | Contents |
|---|---|
| `padchest/`, `padchest_clean/` | flip-bank audits, with and without the answer-inverting rows |
| `balanced_full/` | text-only audit on the 17,144-question balanced bank |
| `gate_balanced_full/` | the vision-ablation gate at full scale |
| `gate_grounded_pilot/` | the same gate on 795 images, kept for the scaling comparison |
| `jlens_faithfulness/` | lens fidelity and the attribution-patching regression |
| `ablation_grounded/`, `ablation_nogrounding/` | the grounding-token ablation |
| `jlens_s0` … `jlens_s7` | eight seeds of the Jacobian lens |
| `sae_s0` … `sae_s7` | eight seeds of the sparse autoencoder |
| `experiment_e/` | rank-1 causal patching on the balanced bank |
| `gate_s0` … `gate_s7` | eight seeds of the gate |

Reproduce with:

```bash
python scripts/data/build_balanced_index.py \
    --questions-csv dataset/padchest/padchest_questions.csv \
    --images-dir cache/padchest_images \
    --out data/index_padchest_balanced_full.json

NANO_INDEX=data/index_padchest_balanced_full.json \
    bash scripts/run/run_balanced_suite.sh
```

## 8. What this does not cover

The MIMIC question metadata was not available, so every new run here is PadChest only.
The published SAE and Jacobian-lens numbers come from the 1,841-question index, which is
861 PadChest questions plus roughly 980 MIMIC ones, and that index could not be rebuilt.
The seed results in section 5 are therefore error bars on the balanced bank, not on the
published setting. They show the effect survives reseeding on a harder benchmark, which is
a different claim and worth stating as one.

The gate was measured on baby-MedGemma. Whether the vision-ablation delta transfers to
MedGemma-4B is untested, and the scaled grounded model with its 66,546 images is the right
place to ask.
