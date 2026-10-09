#!/bin/bash
# Sample quality vs. number of sampling steps, conditional against unconditional: the appendix
# ablation characterising the base model that the gates then act on. One scoring pass computes
# IgLM naturalness, FPD / C2ST / precision / recall, and real-IgBLAST confidence.
#
#   bash run/sample_sweep.sh
#
# Samples: logs/Figures/<run>/conditional/, metrics: logs/Results/<run>/conditional_results.csv
set -euo pipefail

# Edit this one line to switch models. Both released runs were trained for 200k steps, so both
# use checkpoint_20.
# Every setting below can also be overridden from the environment, e.g. NOISE=masking bash run/sample_sweep.sh
NOISE=${NOISE:-uniform}         # uniform | masking
case $NOISE in
    uniform|masking) ;;
    *) echo "NOISE must be 'uniform' or 'masking', got '$NOISE'" >&2; exit 1 ;;
esac
CKPT=20
RUN_NAME=oas95_igblast_human_35M_${NOISE}
REFERENCE_CSV=data/oas95_igblast_human/train.csv

# sampling.classes is deliberately left unset, so labels are drawn from the reference pool's own
# V-gene-family frequencies rather than an even 1/7 share. guid_scale alone separates the two
# series: 0.0 makes the drawn label a no-op (the UNCONDITIONAL baseline), 1.0 lets it drive
# generation (the CONDITIONAL model). Same checkpoint, sampler and label distribution otherwise.
python -m scripts.sample_sequences \
    'model=[amplify,35M,35M_vgene]' \
    trainer=[base,ddlm] \
    trainer.train.max_length=256 \
    trainer.train.diffusion_type=$NOISE \
    trainer.train.scheduler_type=linear \
    sampling.run_name="$RUN_NAME" \
    "sampling.to_evaluate_checkpoints=[$CKPT]" \
    sampling.label_reference_csv=$REFERENCE_CSV \
    sampling.n_per_class=4096 \
    'sampling.guid_scales=[0.0,1.0]' \
    'sampling.num_steps_list=[4,8,16,32,64,128,256]'

python -m scripts.metrics_sequences \
    --run_name "$RUN_NAME" \
    --to_evaluate_checkpoints "$CKPT" \
    --label_reference_csv $REFERENCE_CSV \
    --max_ref_seqs 20000

# ----------------------------------------------------------------------------------------------
# RUN THIS FOR BOTH NOISE TYPES, for the masking-vs-uniform panel. Both are compared at
# checkpoint_20, i.e. the same 200k-step training budget.
#
# PER-CLASS sweeps instead of natural-frequency mode -- n_per_class then means what it says, and
# the output folders are named class_<c>_gs_... rather than class_natural_gs_...:
#   'sampling.classes=[0,1,2,3,4,5,6]' sampling.n_per_class=256
#
# CFG ABOVE 1 amplifies the conditional signal past the plain conditional model:
#   'sampling.guid_scales=[0.0,1.0,2.0,4.0]'
#
# A quick smoke test (a couple of minutes):
#   sampling.n_per_class=32 'sampling.num_steps_list=[32]'
#
# --use_igbert embeds with the antibody-specific IgBert instead of the default small ESM2: more
# sensitive to "does this look like a real antibody", at a large speed cost.
# --skip_precision_recall drops the two expensive columns (see run/gates.sh).
