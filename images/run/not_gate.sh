#!/bin/bash
# NOT gate: avoid generating one digit with posterior-driven (dynamic) negative guidance.
# Sweeps all combinations of the posterior temperature tau and the initial guidance scale, then
# computes FID / precision / recall and the violation rate of a pretrained ViT classifier.
#
#   bash run/not_gate.sh
#
# Samples: logs/Figures/<run>/not_gate/, metrics: logs/Results/<run>/

# Every setting below can also be overridden from the environment, e.g. NOISE=masking bash run/not_gate.sh
NOISE=${NOISE:-uniform}                                 # uniform | masking
RUN_NAME=MNIST_dit_xs_bs512_${NOISE}_ema
UNCOND_RUN_NAME=MNIST_dit_xs_bs512_${NOISE}_ema_uncond

NOT_CLASS=${NOT_CLASS:-0}                           # digit to avoid
TEMPS=${TEMPS:-"[0.01,0.1]"}                        # posterior temperature tau
INIT_GUIDS=${INIT_GUIDS:-"[0.25,0.5,1.0,2.0]"}      # initial guidance scale

# noise.scheduler_type is the noise schedule used for *sampling* (cosine, 128 steps).
python -m scripts.sample_not_gate \
    sampling.run_name=$RUN_NAME \
    sampling.uncond_run_name=$UNCOND_RUN_NAME \
    noise.type=$NOISE \
    noise.scheduler_type=cosine \
    sampling.num_steps=128 \
    sampling.not_class=$NOT_CLASS \
    "sampling.not_temps=$TEMPS" \
    "sampling.not_initial_guid_scales=$INIT_GUIDS" \
    sampling.batch_size=256 \
    sampling.num_batches=16

# SKIP_METRICS=1 samples only: the metrics CSV is read-modify-written, so when several sampling
# jobs of the same run are launched in parallel, score once afterwards instead.
if [[ "${SKIP_METRICS:-0}" != 1 ]]; then
python -m scripts.metrics_not_gate --run_name $RUN_NAME --dataset_name mnist \
    --to_evaluate_checkpoints 200   # add --force to recompute settings already in the CSV
fi

# ----------------------------------------------------------------------------------------------
# Paper results: NOT_CLASS = 0, 1 and 2, each for both noise types, with the same two grids per class
# (run the script once per grid):
#   TEMPS="[0.01,0.025,0.05,0.1,0.2]"  INIT_GUIDS="[0.1,0.2,0.4,0.6,0.8,1.0,1.4,2.2]"
#   TEMPS="[0.01]"                     INIT_GUIDS="[0.05,0.15,0.25,0.3,0.5,0.7,0.9,1.2,1.6,1.8,2.0]"
