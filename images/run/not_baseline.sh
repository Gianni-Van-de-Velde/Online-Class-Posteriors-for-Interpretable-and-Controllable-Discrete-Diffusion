#!/bin/bash
# Baseline for the NOT gate: static negative classifier-free guidance with a constant scale lambda,
# swept over lambda, then FID / precision / recall and ViT violation rate.
#
#   bash run/not_baseline.sh
#
# Samples: logs/Figures/<run>/not_gate/, metrics: logs/Results/<run>/

# Every setting below can also be overridden from the environment, e.g. NOISE=masking bash run/not_baseline.sh
NOISE=${NOISE:-uniform}                                 # uniform | masking
RUN_NAME=MNIST_dit_xs_bs512_${NOISE}_ema
UNCOND_RUN_NAME=MNIST_dit_xs_bs512_${NOISE}_ema_uncond

NOT_CLASS=${NOT_CLASS:-0}                           # digit to avoid
LAMS=${LAMS:-"[0.1,0.2,0.4,0.8,1.6]"}               # negative guidance scale lambda

# noise.scheduler_type is the noise schedule used for *sampling* (cosine, 128 steps).
python -m scripts.sample_not_gate \
    sampling.run_name=$RUN_NAME \
    sampling.uncond_run_name=$UNCOND_RUN_NAME \
    noise.type=$NOISE \
    noise.scheduler_type=cosine \
    sampling.num_steps=128 \
    sampling.not_class=$NOT_CLASS \
    sampling.not_baseline=true \
    "sampling.not_baseline_lams=$LAMS" \
    sampling.batch_size=256 \
    sampling.num_batches=16

# SKIP_METRICS=1 samples only: the metrics CSV is read-modify-written, so when several sampling
# jobs of the same run are launched in parallel, score once afterwards instead.
if [[ "${SKIP_METRICS:-0}" != 1 ]]; then
python -m scripts.metrics_not_gate --run_name $RUN_NAME --dataset_name mnist \
    --to_evaluate_checkpoints 200   # add --force to recompute settings already in the CSV
fi

# ----------------------------------------------------------------------------------------------
# Paper results (both noise types):
#   NOT_CLASS=0  LAMS="[0.025,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.8,1.0,1.5,2.0]"
#   NOT_CLASS=1  LAMS="[0.01,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.3,0.4,0.6,0.8,1.0,1.5,2.0]"
#   NOT_CLASS=2  LAMS="[0.025,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.8,1.0,1.5,2.0]"
