# baby-MedGemma

A small, **MedGemma-faithful probe** for causally isolating *where paraphrase
sensitivity in medical vision-language models originates*. It pairs a frozen
`google/medsiglip-448` vision encoder with a small **Gemma-3 decoder** (rotary
position embeddings, RMSNorm, grouped-query attention, prefix-fused image
tokens), so a mechanism found here transfers to MedGemma-4B, while letting us run
the one experiment the frozen deployed model cannot support: **varying the
training-phrasing distribution one factor at a time.**

- Weights and feature cache: **[huggingface.co/saillab/babymedgemma](https://huggingface.co/saillab/babymedgemma)**
- Interactive write-up: **[bineshkumar.me/phd-thesis/causality](https://bineshkumar.me/phd-thesis/causality/)**
- Design document: [`docs/tiny_vlm_psf_isolation.md`](docs/tiny_vlm_psf_isolation.md)

Companion to the PhD dissertation *Paraphrase Sensitivity in Medical
Vision-Language Models* (Thrust 3, mechanism and mitigation).

> **Scope.** This is a controlled research probe, **not a general medical VQA
> model**. It answers binary presence questions about 14 chest findings, using
> MedGemma's SentencePiece vocabulary pruned to the 141 pieces this corpus needs.
> It supports a *sufficiency* claim about the cause of paraphrase sensitivity, not
> proof that the deployed MedGemma-4B has this exact origin. Not a medical device;
> not for clinical use.
> The experiments below state exactly what it does and does not establish.

## The question

Medical vision-language models often change their yes/no answer when a clinical
question is rephrased without changing its meaning. On a deployed model you can
find *where* that flip is decided (layer 16 on MedGemma-4B) and *what* feature
carries it (a layer-17 register gate), but not *why* it exists, because the
model's data, architecture, and objective are frozen together. To ask *why*, you
have to intervene on one of them and read the effect. That is what this probe is
for.

## The probe

```
chest X-ray ──► MedSigLIP at 896px (frozen, 429M) ──► 4096 patches pooled to 256 tokens
                                                        │  prepended inline (prefix fusion)
question (paraphrase) ──► token embeddings ─────────────┤
                                                        ▼
                                        Gemma-3 decoder (trained, 14.1M)
                                        RoPE · RMSNorm · GQA (6 Q / 2 KV) · causal mask
                                                        ▼
                                yes / no read from the tied Gemma-3 LM head
```

Trained on 86,288 per-finding-balanced presence questions from **NIH ChestX-ray14
and PadChest**, with **MIMIC-CXR and VinDr-CXR held out entirely**, each question
carrying 48 register-tagged paraphrases. It reaches 74.8% accuracy and, because
every split is balanced per finding, **exactly 50.0% with the image removed**, so
all of its skill is visual. It transfers to both unseen hospitals (AUC 0.743 MIMIC,
0.756 VinDr) above MedSigLIP's own zero-shot 0.734. Two properties make it useful:

1. **Language-side by construction.** The encoder returns *identical* features for
   every paraphrase, so any answer that changes across paraphrases is in the
   decoder, not the image. Guaranteed by the setup, not measured.
2. **The origin is a knob.** We can vary the training-phrasing distribution with
   everything else held fixed, turning a difference in the flip rate into an
   *identified cause*.

## Fidelity to MedGemma-1.5

The probe follows the MedGemma-1.5 technical report
([arXiv:2604.05081](https://arxiv.org/pdf/2604.05081)) on every architectural
choice that decides *how* a paraphrase reaches the answer, and simplifies only the
axes of scale and training that a controlled causal experiment needs to strip
away. The first table is why a mechanism found here is expected to transfer; the
second is why this is a probe and not a deployable model.

**Matched to the deployment stack**

| Principle | MedGemma-1.5 | baby-Gemma |
|---|---|---|
| Vision encoder | MedSigLIP (400M SigLIP), frozen | MedSigLIP-448, frozen |
| Input resolution | 896 x 896 | 896 x 896 (position embeddings interpolated) |
| Image tokens per image | 256 (pooled) | 4096 patches average-pooled to 256 |
| Vision-to-text fusion | soft image tokens projected inline | multilayer-perceptron projector, prepended inline |
| Decoder family | Gemma-3 | Gemma-3 blocks: RoPE, RMSNorm, grouped-query attention, GeGLU, QK-norm |
| yes/no readout | generated as text through the LM head | yes/no token logits from the tied LM head |

**Intentionally simplified for the experiment**

| Axis | MedGemma-1.5 | baby-Gemma | Why |
|---|---|---|---|
| Decoder scale | 4B parameters, pretrained | 14.1M, trained from scratch | so the training-phrasing distribution is a controllable input, not a frozen prior |
| Training objective | supervised fine-tuning, distillation, reinforcement learning | supervised, validation early-stopped | keeps the only manipulated variable the phrasing distribution |
| Vocabulary | full Gemma tokenizer (262,144) | **same tokenizer**, pruned to the 141 pieces this corpus uses (0.4% of the parameters) | identical segmentation to MedGemma; unseen words decompose into pieces rather than becoming padding |
| Local attention | 5:1 local/global sliding window | one window over the whole short sequence | the 257+20 sequence is shorter than any window, so the pattern is inert here |

## The framework: five experiments, A to E

The five experiments isolate the origin in one line each:

| | Experiment | Question it answers |
|---|---|---|
| **A** | Data provenance | Does the *training data* cause the sensitivity? |
| **B** | Divergence trajectory | *Where* in the network does the disagreement emerge? |
| **C** | Causal patching | *What* decides the flip, and is it low-rank? |
| **D** | Architecture | Is it a property of model *depth*? |
| **E** | Grounding | Is it caused by *weak image grounding*? |

A establishes the cause; B and C localize and confirm the mechanism; D and E rule
out the two obvious alternatives. A, C and the threshold checks were re-run on the
scaled model described above; **B, D and E are carried over from the earlier
1,841-question probe** (retired; its results remain under `results_gemma/` and the
scripts are in `scripts/legacy/`). Two corroborations (an unsupervised feature and
a lens) and a zero-shot transfer test close it out. All figures are baby-Gemma
over 8 seeds (`results_transfer_grid/`).

### A. The cause: data provenance

**What we did.** Train the same probe under three phrasing regimes that differ
*only* in the questions it sees: *canonical* (one fixed phrasing), *augmented*
(every paraphrase), and *adversarial* (register tied to the answer). Because only
the training-phrasing distribution changes, a difference in the flip rate is an
identified cause.

| Regime | Trained on | Flip rate | Flip rate, **unseen phrasings** |
|---|---|---|---|
| Augmented | every paraphrase | **4.8%** | **26.6%** |
| Canonical | one fixed phrasing | 67.1% | 65.9% |
| Adversarial | register tied to the answer | 88.4% | 87.4% |

Augmented separates from both narrow regimes at the maximum effect size
(Mann-Whitney U p = 1.6e-4, Cliff's delta = 1.00, 8 seeds each), and the two narrow
regimes now separate from each other as well (p = 3.1e-4, delta = 0.97): a phrasing
shortcut is a **distinct and larger harm** than narrow coverage alone. Accuracy
follows the same ordering (75.3% / 66.8% / 58.1%), so the shortcut damages diagnosis
and not only consistency. Text-only accuracy is 50.0-50.3% in every regime.

The second column is the honest one. A model trained on every paraphrase has seen
the phrasings it is tested on, so part of a low flip rate is familiarity. Training
on 24 phrasings and scoring **only on the 24 withheld** (6 of each phenomenon per
side), augmented rises 4.8% -> 26.6% but still flips less than half as often as
canonical, delta = 1.00 (p = 2.2e-3). **Broad coverage generalises to wording never
seen in training** -- which is what the mitigation claim actually needs.

**What it proves.** The training-phrasing distribution is *sufficient* to produce
and to remove paraphrase sensitivity, and broad coverage is the single largest
lever against it. Correlating register with the answer adds nothing beyond a
single phrasing, so the reproducible finding is the augmentation lever, not a
taxonomy of flip types.

### B. Where it emerges: divergence trajectory

Within-cluster representation dispersion couples to the flip from the earliest
layers (point-biserial near 0.76 at the input for natural flips, 0.88 for adversarial),
and lexical substitution and scope shifts drive the most disagreement. **The
disagreement is present early and carried by the wording, not seeded in the
image.**

### C. The mechanism: causal patching

For a flipped question, transplant the answer position along a single rank-1
direction (the difference between a phrasing answered one way and one answered the
other) at one layer at a time. It restores the flipped answer with net recovery
0.98-1.00 across layers 0 to 4 for adversarial and 0.77 rising to 1.00 for natural
flips, while a norm-matched random direction (0.00-0.02) and a non-flip-cluster
control (disruption 0.000-0.006) leave the answer unchanged. This holds for the *naturally occurring* flips of
the augmented regime (59 per seed), not only the injected adversarial ones (60
per seed). **The flip is a low-rank, language-side, readout-stage direction,
decided in the early layers, and not an artifact of the adversarial construction.**

### D. Ruling out architecture

Across decoder depths 2, 4, 6, 8 the flip rate stays within 9.1% to 10.2% at
constant accuracy. **Depth is not the driver.**

### E. Ruling out weak grounding

Weakening the visual pathway by dropping vision tokens gives a non-monotonic
flip-rate curve. **Inconclusive; no weak-grounding claim is drawn from it.**

## Corroboration: an unsupervised feature, and a lens

Two independent methods that make different assumptions agree with C.

**A sparse autoencoder (`sae.py`)** factors the residual stream at the answer
position into a few active features, each a direction; we ask whether any feature
matches the causal flip direction from C.

![What the sparse autoencoder tests](figures/sae_concept.png)

On the retired 1,841-question probe (`results_gemma/sae_gemma/`, 231 paraphrase
clusters, d = 384), an unsupervised feature aligns with the causal flip direction
at **|cosine| 0.74**, ahead of principal-component analysis at 0.52. Both are
*best-of-many* statistics, so the reference has to be selected the same way: the
best of 2,048 random directions reaches 0.18 by chance (95th percentile 0.21) and
the best of 20 reaches 0.11, against a single random direction's 0.04. Each clears
its matched null by roughly four times. That same feature independently predicts
flips (point-biserial 0.42, p = 2e-11); it was selected by alignment with the
causal direction rather than by its own flip correlation, so that is one test
rather than a search over the dictionary.

On the scaled grounded model (`results_transfer/sae/`, 73,500 evaluations) the
alignment is weaker but survives: |cosine| 0.59 against the same 0.18 null, and
point-biserial 0.13 against a matched max-of-2,048 null of 0.09. The flip axis is
unsupervised-recoverable on both models, not an artifact of the supervised
difference-of-means.

![An unsupervised feature recovers the causal flip axis](figures/sae_alignment.png)

The method, the corrected nulls, and eleven implementation caveats with the fix for
each are worked through interactively in
[`docs/sae_explainer.html`](docs/sae_explainer.html).

**A Jacobian lens (`jlens.py`)** reads the yes/no margin each layer is disposed to
produce — the average input-output Jacobian, differentiated through the remaining
layers rather than assuming them away.

![The idea behind the Jacobian lens, drawn schematically](figures/jlens_concept.png)

*The figure above is a schematic of the idea, not plotted data.* The measurement
below is the result, and its shape differs: the phrasings never track together.

On the retired 1,841-question probe (`results_gemma/jlens_gemma/`, 231 clusters),
flipping clusters' lens margins diverge **8.9x** more than stable ones, already at
layer 0 and flat across all six layers, with a divergence-vs-flip correlation of
**0.81**. On the scaled grounded model (`results_transfer/jlens/`, 1,500 clusters)
the same statistics are **2.2x** and **0.26**. The lens and the rank-1 causal patch
agree that the flip is decided at or before the first decoder layer, but neither
localises it further: both sit at their ceiling from layer 0 onward.

![The lens splits flipping from stable paraphrases](figures/jlens_divergence.png)

The lens is the first-order approximation of that same patch, not an independent
method, so their agreement is a consistency check rather than corroboration; the
independent leg is the unsupervised SAE direction above. The method, its lineage
from the logit and tuned lenses, and eleven implementation caveats with the fix for
each are worked through interactively in
[`docs/jlens_explainer.html`](docs/jlens_explainer.html).

## Why binary accuracy cannot tell a blind model from a seeing one

The probe above scored **0.50 accuracy on unseen NIH radiographs**, which reads as
total failure. Scoring the same models by the **AUC of their yes/no margin** instead
shows what accuracy was hiding (16 seeds, balanced 50/50 set):

| model | accuracy | AUC |
|---|---|---|
| no grounding token | 0.50 | **0.500** (genuinely blind, 16/16 seeds) |
| grounding token **shuffled** across images | 0.52 | 0.502 |
| grounding token, real | 0.53 | **0.604** |

All three score ~0.50 accuracy. Their AUCs are chance, chance, and clearly-not-chance.
A binary yes/no readout collapses a continuous margin onto one decision threshold per
finding, and that threshold is learned from the training distribution's prevalence;
under a distribution shift the whole margin distribution moves and every image lands
on one side. So the model answers "yes" to all cardiomegaly and "no" to all masses at
exactly 50% accuracy, while ranking both correctly.

MedSigLIP itself pays this tax: **AUC 0.734 but accuracy 0.681** on the same task.
**Report AUC.** Accuracy alone cannot detect whether a medical VLM is using the image,
which is the metric the field mostly uses.

The in-distribution flip-rate results are *not* affected by this. A threshold sweep
holds the augmented < canonical ordering at every non-degenerate offset, and a
threshold-free statistic (within-cluster margin dispersion over between-cluster
dispersion) reproduces it at maximal effect size: augmented 0.023, canonical 0.498,
adversarial 1.362, Cliff's delta = -1.00 against augmented (p = 1.6e-4).

## Scaling up: a grounded model that transfers (`build_transfer_index.py`)

The probe is deliberately small. Given the diagnosis above, three changes make it
actually read radiographs: **per-finding answer balancing** (so the question text
predicts nothing and the only way to reduce loss is to look), the **grounding token**,
and **scale** (107k questions over 66,546 images from the full ChestX-ray14). Trained
on NIH + PadChest with **MIMIC and VinDr held out entirely**:

| split | n | accuracy | AUC | text-only floor | flip |
|---|---|---|---|---|---|
| in-distribution (held-out images) | 15,112 | 0.748 | 0.827 | **0.500** | 0.061 |
| **MIMIC (unseen hospital)** | 450 | 0.671 | **0.743** | **0.500** | 0.049 |
| **VinDr (unseen hospital)** | 5,478 | 0.686 | **0.756** | **0.500** | 0.060 |

Both held-out hospitals exceed MedSigLIP's own zero-shot ceiling (0.734), against the
small probe's 0.500. The text-only floor is exactly 0.500 everywhere by construction,
so every point above it is earned from the image: **+24.8 pp of visual skill**, versus
+19.4 pp for the 88.2%-accurate original whose blind floor was 68.8%.

Scale alone was not the fix. An earlier 10k scale-up was abandoned as "stuck at
chance" — it was scored by accuracy, and it lacked both the balancing and the
grounding token. All three were needed.

Per finding, pneumothorax transfers essentially unchanged (0.899 -> 0.903 on VinDr).
**Pleural thickening inverts on VinDr (AUC 0.332, below chance)**, almost certainly a
label-definition mismatch between the two datasets, and is not trustworthy.

## The claim, stated exactly

The origin of paraphrase sensitivity is the **training-phrasing distribution**,
executed as a **low-rank direction in the early language layers** that read a
fixed visual representation. The fix follows: paraphrase augmentation is the lever
(67.1% down to 4.8%, and to 26.6% on wording withheld from training), and a
targeted low-rank edit at those layers is the
efficient parametric fix, which is why a layers-15-to-19 low-rank adaptation
reduces flips on the deployed model while full fine-tuning does not.

## What this does NOT prove

- **It is a controlled probe, not the deployed model.** The result is a
  *sufficiency* claim plus a localization, not proof that MedGemma-4B's paraphrase
  sensitivity has this exact origin. Its absolute layer index (early, layers 0-4)
  is not comparable to the deployed model's layer-16 commit; only the qualitative
  account transfers.
- **It manipulates adaptation-stage data, not pretraining provenance.**
- **The paraphrases are a bank of 48 templates across four phenomena,** not free
  clinician writing. The held-out-phrasing result establishes generalization to
  unseen *templates*, not to arbitrary unseen language.
- **Per-finding balancing removes the answer prior by design.** That is what lets
  a flip be attributed to wording rather than a base rate, but it also makes the
  probe unrepresentative of deployment, where prevalence is skewed and a model can
  be right for the wrong reason.
- **It is a narrow model, not a medical VQA system:** 14 findings, binary
  questions, loadable and runnable for inspection and reproduction only. Not for
  clinical use.

## Why it still matters: triangulation

The probe is one leg of a three-method argument that no single method licenses. On
MedGemma-4B, a lens-free residual patch and the Jacobian lens put the answer
commit at layer 16, and a GemmaScope sparse autoencoder finds a layer-17 register
gate (Feature 3818). The probe adds the one thing those cannot: a controlled
origin. The three agree that the flip is language-side, decided in a narrow
early-to-middle band, and set by the training-phrasing distribution.

## Layout

An installable package (`pip install -e .`) under `src/babygemma/`, with entry-point
scripts grouped by role under `scripts/`. The scripts import the package rather than
their siblings, and resolve data paths through `babygemma.paths`, so nothing depends on
where a file sits in the tree.

```
src/babygemma/            the importable library
  paths.py                repository-root-anchored cache / data / results paths
  vision.py               frozen MedSigLIP-448 (896px, 4096 -> 256 pooled tokens)
  data_index.py           the VQA index (NANO_INDEX selects which one)
  nih.py                  NIH ChestX-ray14 question and label builders
  dataset.py              the three training regimes + paraphrase clusters
  tokenizer.py            MedGemma SentencePiece pruned to the corpus
  templates.py            48-paraphrase bank (meaning-preserving; negation separate)
  gemma_model.py          BabyGemmaVLM: Gemma-3 decoder + prefix fusion + grounding token
  encoders.py             MedSigLIP attention-pooled embedding (the grounding token)
  metrics.py              flip rate, accuracy, grounding gap, dispersion
  training.py             train_model()

scripts/
  data/          download_nih, build_transfer_index, build_scaled_index,
                 precompute_features, precompute_pooled, encode_shard, merge_shards
  train/         train.py (CLI), run_all_gpus.py (grid scheduler)
  experiments/   experiment_a (divergence), experiment_e (rank-1 patching), sae, jlens,
                 flip_threshold_robustness
  analysis/      eval_transfer, eval_scaled
  hf/            modeling_babymedgemma.py (self-contained wrapper), convert_to_hf.py
  run/           shell drivers that chain the above into full runs
  legacy/        scripts of the retired 1,841-question probe (results under results_gemma/)
```

Results directories are documented in [`RESULTS.md`](RESULTS.md). Heavy artifacts
(`**/model.pt`, `cache/`) are on Hugging Face, not in git.
(`**/model.pt`, `cache/`) are on Hugging Face, not in git.


The model checkpoints (`**/model.pt`, ~4 GB) and the MedSigLIP feature cache
(`cache/medsiglip_feats.pt`, ~3 GB) are on Hugging Face, not in git:
**[saillab/babymedgemma](https://huggingface.co/saillab/babymedgemma)**.

## Load with transformers

Packaged as a custom `transformers` model, loadable with `trust_remote_code`:

```python
import torch
from transformers import AutoModel
from PIL import Image

model = AutoModel.from_pretrained("saillab/babymedgemma", trust_remote_code=True).eval()
input_ids, ans_pos = model.encode_question("is there pleural effusion?")
vision_features, ground = model.encode_images([Image.open("cxr.png").convert("RGB")])
logits = model(input_ids=input_ids, vision_features=vision_features,
               ground=ground, ans_pos=ans_pos).logits
print(model.config.id2label[int(logits.argmax(-1))])   # "yes" or "no"
```

`encode_images` needs gated access to `google/medsiglip-448`. The model uses MedGemma's
own SentencePiece tokenizer pruned to 141 pieces, so an unseen word decomposes into
pieces rather than silently becoming padding.

## Reproduce

```bash
# 0. install the package (editable) + dependencies. Needs gated google/medsiglip-448.
pip install -e .

# 1. data: download NIH ChestX-ray14, build the transfer index, cache both encodings
python scripts/data/download_nih.py
python scripts/data/build_transfer_index.py                    # -> data/index_transfer.json
python scripts/data/precompute_features.py                     # 896 patch tokens
python scripts/data/precompute_pooled.py                       # attention-pooled grounding token

# 2. one run (env vars select the index, caches, and tokenizer; see scripts/run/)
NANO_INDEX=$PWD/data/index_transfer.json NANO_GEMMA_TOK=1 NANO_MAXLEN=20 \
NANO_FEATS=$PWD/cache/transfer_feats.pt NANO_POOLED=$PWD/cache/transfer_pooled.pt \
  CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --regime augmented --seed 0 --grounding-token

# 3. the full provenance grid across 8 GPUs, and the follow-on experiments
bash scripts/run/run_gridA.sh
bash scripts/run/run_all_experiments.sh
```

Each framework letter maps to a grid tag and a result file:

| Framework | Script | Result file (latest model) |
|---|---|---|
| A Data provenance | `run_all_gpus.py` tag B | `results_transfer_grid/B/*/result.json` |
| A' Held-out phrasing | `run_all_gpus.py` (NANO_PARA_HELDOUT) | `results_transfer_heldout/B/*/result.json` |
| C Causal patching | `experiment_e.py` | `results_transfer_patch/*/experiment_e.json` |
| Threshold robustness | `flip_threshold_robustness.py` | `results_transfer_grid/dispersion.json` |
| Transfer to unseen hospitals | `eval_transfer.py` | `results_transfer/eval_transfer.json` |
| SAE, Jacobian lens | `sae.py`, `jlens.py` | `results_transfer/{sae,jlens}/*.json` |

The 1,841-question probe's experiments B, D and E, and its NIH diagnosis, are under
`results_gemma/` with their scripts in `scripts/legacy/`.

## Citation

> Sadanandan, B. (2026). *Paraphrase Sensitivity in Medical Vision-Language
> Models: Measurement, Mechanisms, Mitigation, and Deployment Safety.*
> Unpublished doctoral dissertation, University of New Haven.

## License and data

Code is released for research use. The probe is trained on NIH ChestX-ray14 and
PadChest, with MIMIC-CXR and VinDr-CXR held out entirely for transfer evaluation;
use of the derived features and checkpoints is subject to those datasets' data-use
agreements. Not a medical device and not for clinical use.
