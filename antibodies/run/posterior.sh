#!/bin/bash
# Online class posterior during unconditional generation: agreement between the posterior's
# argmax and real IgBLAST V-gene classification, as a function of time, plus example
# trajectories. Costs num_classes+1 = 8 forward passes per sampling step.
#
#   bash run/posterior.sh
#
# Results and figures: logs/Figures/<run>/posterior_unconditional/ckpt_<k>/n_<N>_steps_<s>_sched_linear/
set -euo pipefail

# Edit this one line to switch models. Both released runs were trained for 200k steps, so both
# use checkpoint_20.
# Every setting below can also be overridden from the environment, e.g. NOISE=masking bash run/posterior.sh
NOISE=${NOISE:-uniform}         # uniform | masking
case $NOISE in
    uniform|masking) ;;
    *) echo "NOISE must be 'uniform' or 'masking', got '$NOISE'" >&2; exit 1 ;;
esac
CKPT=20
RUN_NAME=oas95_igblast_human_35M_${NOISE}

# Sampling is purely unconditional: every generative step is driven by the unconditional model
# only, so the tracked posterior is a read-out rather than something steering generation. That
# is the point -- it is then compared against what IgBLAST says was actually generated.
python -m scripts.posterior_classifier_agreement \
    'model=[amplify,35M,35M_vgene]' \
    trainer=[base,ddlm] \
    trainer.train.max_length=256 \
    trainer.train.diffusion_type=$NOISE \
    trainer.train.scheduler_type=linear \
    sampling.run_name="$RUN_NAME" \
    "sampling.to_evaluate_checkpoints=[$CKPT]" \
    sampling.label_reference_csv=data/oas95_igblast_human/train.csv \
    sampling.max_ref_seqs=20000 \
    sampling.posterior_num_samples=2048 \
    sampling.num_steps=64 \
    sampling.batch_size=128 \
    "$@"

# ----------------------------------------------------------------------------------------------
# RUN THIS FOR BOTH NOISE TYPES: the paper's figures overlay masking against uniform. Keep
# posterior_num_samples and num_steps the same across the two -- the plot builders pick the run
# with the most samples per noise type, so a mismatched n silently compares different sizes.
#
# A cheaper run while checking the plumbing (seconds rather than ~an hour):
#   sampling.posterior_num_samples=64 sampling.num_steps=20 sampling.batch_size=32
#
# The AUC / ROC / calibration analyses are all post-hoc on the cached per-step log-ratios, so a
# temperature sweep costs no extra sampling: it is already in the output as
# auc_vs_time_softmax_tau*.csv alongside the unnormalised auc_vs_time_unnorm.csv.
