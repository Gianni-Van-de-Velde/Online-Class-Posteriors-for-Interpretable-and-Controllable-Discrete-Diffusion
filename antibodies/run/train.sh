#!/bin/bash
# Train the class-conditional AMPLIFY-35M antibody model of the paper: 200k steps, effective
# batch size 256, label dropout 0.15 so one model serves as both the conditional and the
# unconditional network for CFG and for the gates.
#
#   bash run/train.sh
#
# Checkpoints: logs/<run>/checkpoints/checkpoint_<step/save_steps>/, W&B: logs/<run>/wandb/
# Extra Hydra overrides can be appended, e.g. bash run/train.sh trainer.max_steps=1000
set -euo pipefail

# Edit this one line to switch models. Both released runs were trained for 200k steps.
NOISE=uniform                       # uniform | masking
case $NOISE in
    uniform|masking) ;;
    *) echo "NOISE must be 'uniform' or 'masking', got '$NOISE'" >&2; exit 1 ;;
esac
MAX_STEPS=200000                    # final checkpoint: checkpoint_20
RUN_NAME=oas95_igblast_human_35M_${NOISE}

# Effective batch 256 = per_device_batch_size 128 x gradient_accumulation_steps 2, on one GPU.
# max_length=256 rather than 512: nothing in this dataset comes near 512, and the collator pads
# to max_length while attention cost scales with the padded length squared.
accelerate launch --config_file=conf/accelerate.yaml --num_processes=1 \
    --mixed_precision=bf16 --gradient_clipping=1.0 \
    scripts/train.py \
    hydra.run.dir=logs/$RUN_NAME \
    wandb.dir=logs/$RUN_NAME \
    wandb.name=$RUN_NAME \
    dataset=oas95_igblast_human_train \
    'model=[amplify,35M,35M_vgene]' \
    'trainer=[base,ddlm]' \
    trainer.dir=logs/$RUN_NAME \
    trainer.resume=True \
    trainer.train.max_length=256 \
    trainer.validation.max_length=256 \
    trainer.train.diffusion_type=$NOISE \
    trainer.validation.diffusion_type=$NOISE \
    trainer.train.scheduler_type=linear \
    trainer.validation.scheduler_type=linear \
    trainer.train.per_device_batch_size=128 \
    trainer.validation.per_device_batch_size=128 \
    trainer.gradient_accumulation_steps=2 \
    trainer.train.label_dropout_prob=0.15 \
    trainer.save_steps=10000 \
    trainer.eval_steps=10000 \
    trainer.max_steps=$MAX_STEPS \
    optimizer.lr=1e-4 \
    "$@"

# ----------------------------------------------------------------------------------------------
# EARLY-TRAINING LOSS CURVES. The run above first validates at step 10000, by which point almost
# the whole loss drop has happened (3.11 at step 100 -> 0.54 at 1000 -> 0.363 at 200000). This
# replays the first 10k steps evaluating every 100 instead -- a faithful replay, since the LR
# schedule comes from conf/scheduler/ and does not depend on max_steps:
#   RUN_NAME=oas95_igblast_human_35M_${NOISE}_denseval
#   dataset=oas95_igblast_human_train_denseval \
#   trainer.eval_steps=100 \
#   trainer.max_steps=10000 \
# The denseval dataset config only swaps the validation split for a 20k-row stratified
# subsample, because evaluate() walks the whole validation dataloader with no early break.
#
# With N GPUs: --num_processes=N and gradient_accumulation_steps=2/N holds the effective batch.
# The paper's runs used 2 GPUs with no accumulation, which is the same effective batch and LR.
#
# W&B is offline by default (conf/config.yaml). To sync: wandb.mode=online wandb.entity=<entity>
#
# Neither model is converged: 200k steps at batch 256 is well under one epoch of a 54.3M-sequence
# set.
