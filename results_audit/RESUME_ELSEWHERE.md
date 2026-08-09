# Picking this up on another machine

Everything needed to keep going is either in git or in the Hugging Face bucket
`binesh/tbucket`. Nothing lives only on the RunPod box.

## What is where

| Artifact | Size | Where | Cost to rebuild |
|---|---|---|---|
| All code and all result JSON | small | git, this branch | n/a |
| MedSigLIP patch features, 4,552 images | 2.6 GB | bucket `fliplens_cache/medsiglip_feats.pt` | 28 min on an A6000 |
| MedSigLIP pooled embeddings | 11 MB | bucket `fliplens_cache/medsiglip_pooled.pt` | 18 min on an A6000 |
| Balanced index, 17,144 questions | 97 MB | bucket `fliplens_cache/` | 10 seconds from the CSV |
| PadChest PNGs, 4,555 images | 36 GB | bucket `padchest/` | download only |

The two caches are the ones worth pulling. With them, no image ever has to be encoded
again, and none of the remaining work needs a big GPU.

## Setup

```bash
git clone git@github.com:thedatasense/FlipLens.git
cd FlipLens && git checkout docs/sae-interactive-explainer
pip install -e . && pip install accelerate sentencepiece protobuf

# bucket access needs huggingface_hub >= 1.0, which conflicts with transformers'
# pin, so keep it in its own venv and use the main environment for the model
python -m venv .hfvenv && .hfvenv/bin/pip install -U huggingface_hub
export HF_TOKEN=...            # rotate the old one first

mkdir -p cache data
.hfvenv/bin/hf buckets cp hf://buckets/binesh/tbucket/fliplens_cache/medsiglip_feats.pt  cache/
.hfvenv/bin/hf buckets cp hf://buckets/binesh/tbucket/fliplens_cache/medsiglip_pooled.pt cache/
.hfvenv/bin/hf buckets cp hf://buckets/binesh/tbucket/fliplens_cache/index_padchest_balanced_full.json data/
```

Then run anything in the suite:

```bash
export NANO_INDEX=$PWD/data/index_padchest_balanced_full.json
export NANO_PARA_SAMPLE=12 NANO_PATIENCE=15 NANO_EVAL_EVERY=150
python scripts/analysis/ablation_gate.py --grounding-token --seed 3 \
    --steps 6000 --out results_audit/gate_s3
```

The tokenizer pulls `google/medgemma-4b-it` from the Hub on first use, so the token
needs access to that gated repo. It's cached after the first run.

## Running on an M1 Max

The trainable model is 14.1M parameters over a 277-token sequence, so a Mac handles it.
The frozen encoder is the only heavy part and its output is already cached.

One change is needed. `src/babygemma/training.py` line 48 reads:

```python
device = device or ("cuda" if torch.cuda.is_available() else "cpu")
```

There's no Metal branch, so a Mac silently falls back to CPU. Add one:

```python
device = device or ("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
```

`vision.py`, `encoders.py` and `nih.py` also hardcode `device="cuda:0"`, but those only
matter for encoding images, which the cache makes unnecessary. Scripts under
`scripts/legacy/` hardcode `.cuda()` and would each need a device argument; none of them
are part of the remaining work.

Expect roughly three to eight times slower per training run than the A6000. A single
`ablation_gate.py` run is minutes, not hours.

## What is left to run

The suite in `scripts/run/run_balanced_suite.sh` was interrupted partway. Completed and
committed:

| Stage | State |
|---|---|
| 1. Lens fidelity and patch validation | done |
| 2. Grounding-token ablation | done |
| 3. jlens across 8 seeds | done |
| 4. SAE across 8 seeds | seeds 0 to 5 done, 6 and 7 outstanding |
| 5. Rank-1 causal patching | not started |
| 6. Ablation gate across 8 seeds | not started |

Stages 4 to 6 are eleven training runs. Each is one `train_model` call at 6,000 steps
with early stopping. To finish only what's missing:

```bash
for s in 6 7; do
  python scripts/experiments/sae.py --grounding-token --seed $s --layer 1 \
      --model-steps 6000 --out results_audit/sae_s$s
done
python scripts/experiments/experiment_e.py --grounding-token --regime augmented \
    --steps 6000 --max-clusters 60 --out results_audit/experiment_e
for s in 0 1 2 3 4 5 6 7; do
  python scripts/analysis/ablation_gate.py --grounding-token --seed $s \
      --steps 6000 --out results_audit/gate_s$s
done
```

If time is short, cut the gate to three seeds. The point of that stage is an interval on
the +7.2 accuracy points, and three seeds give one.

## Then update the summary

`results_audit/README.md` currently says the SAE table is from four seeds and carries no
numbers for stages 5 and 6. Fill those in once the runs land.

## Security

The Hugging Face token used during the RunPod session was pasted into a chat transcript.
Rotate it at https://huggingface.co/settings/tokens before reusing it anywhere.
