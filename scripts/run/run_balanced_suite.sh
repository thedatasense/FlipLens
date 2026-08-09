#!/usr/bin/env bash
# Everything worth running on the per-finding balanced PadChest bank.
#
# The flip bank cannot measure grounding: all 861 questions span 92 findings and no
# finding carries both answers, so a lookup on the finding name scores 861/861 and
# baby-MedGemma scores 1.000 there with every vision token zeroed. The balanced bank
# built by scripts/data/build_balanced_index.py drives that ceiling to 50.0%, which
# makes it the first setting in this repo where "does the model read the radiograph"
# is a question with a measurable answer.
#
# Ordered by how much each result adds to the write-up. Everything here ADDS numbers;
# nothing recomputes a published one.
#
#   bash scripts/run/run_balanced_suite.sh 2>&1 | tee results_audit/suite.log
set -u

export PYTHONPATH=/workspace/FlipLens/src:/workspace/FlipLens/scripts/experiments
export NANO_INDEX=${NANO_INDEX:-/workspace/FlipLens/data/index_padchest_balanced_full.json}
export NANO_PARA_SAMPLE=${NANO_PARA_SAMPLE:-12}
export NANO_PATIENCE=${NANO_PATIENCE:-15}
export NANO_EVAL_EVERY=${NANO_EVAL_EVERY:-150}
STEPS=${STEPS:-6000}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
OUT=results_audit
mkdir -p $OUT

banner() { echo; echo "################ $* ################"; date -u +"%Y-%m-%d %H:%M:%SZ"; }

# Results are committed after every stage, so an interrupted suite still leaves
# everything it finished on the branch.
save() {
    cd /workspace/FlipLens || return
    git add -A results_audit/ >/dev/null 2>&1
    git diff --cached --quiet && { echo "[save] nothing new"; return; }
    git commit -q -m "Balanced-bank suite: $1" \
        -m "Auto-committed by scripts/run/run_balanced_suite.sh. Adds results only." \
        -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
        && git push -q origin HEAD && echo "[save] committed and pushed: $1"
}

# 1. Does the lens predict what the model does, and does it predict the patch?
#    The single biggest hole in the Jacobian-lens chapter: nothing ever checked that
#    the lens reading tracks the true margin, or that its difference across a donor
#    and target predicts the effect of the rank-1 patch it is supposed to corroborate.
banner "1/6  lens fidelity and the attribution-patching validation"
python3 scripts/experiments/jlens_faithfulness.py --grounding-token \
    --model-steps $STEPS --max-pairs 80 --out $OUT/jlens_faithfulness
save "lens fidelity and patch validation"

# 2. The grounding token is the whole reason this task is learnable at all. Show it.
banner "2/6  grounding-token ablation"
for g in "--grounding-token" ""; do
    tag=$([ -n "$g" ] && echo grounded || echo nogrounding)
    python3 scripts/analysis/text_only_audit.py --steps $STEPS $g \
        --out $OUT/ablation_$tag
done
save "grounding-token ablation"

# 3. Error bars. jlens and the SAE were each run once, at seed 0, against 8 to 24
#    seeds everywhere else in the project.
banner "3/6  jlens across seeds"
for s in $SEEDS; do
    python3 scripts/experiments/jlens.py --grounding-token --seed $s \
        --model-steps $STEPS --out $OUT/jlens_s$s
done
save "jlens across seeds"

banner "4/6  sparse autoencoder across seeds"
for s in $SEEDS; do
    python3 scripts/experiments/sae.py --grounding-token --seed $s --layer 1 \
        --model-steps $STEPS --out $OUT/sae_s$s
done
save "sparse autoencoder across seeds"

# 5. Does the causal locus hold when the answer actually needs the image?
banner "5/6  rank-1 causal patching"
python3 scripts/experiments/experiment_e.py --grounding-token --regime augmented \
    --steps $STEPS --max-clusters 60 --out $OUT/experiment_e
save "rank-1 causal patching"

# 6. The gate, across seeds, so the accuracy gain gets an interval.
banner "6/6  ablation gate across seeds"
for s in $SEEDS; do
    python3 scripts/analysis/ablation_gate.py --grounding-token --seed $s \
        --steps $STEPS --out $OUT/gate_s$s
done
save "ablation gate across seeds"

banner "SUITE COMPLETE"
