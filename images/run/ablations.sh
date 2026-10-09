#!/bin/bash
# Model ablations (appendix): FID, precision/recall, C2ST and classifier accuracy as a function of
# the number of sampling steps, the guidance scale and the sampling noise schedule.
#
#   bash run/ablations.sh
#
# Samples: logs/Figures/<run>/samples/, metrics: logs/Results/<run>/

NOISE=uniform                             # uniform | masking
RUN_NAME=MNIST_dit_xs_bs512_uniform_ema   # MNIST_dit_xs_bs512_{uniform,masking}_ema[_uncond]
GUIDANCE_SCALES="0 1 2 3"                 # "0" for the unconditional models (_uncond)

for SCHEDULER in linear sin cosine; do
    for GUID in $GUIDANCE_SCALES; do
        for STEPS in 4 8 16 32 64 128; do
            # noise.scheduler_type is the noise schedule used for *sampling*.
            python -m scripts.sample_images \
                sampling.run_name=$RUN_NAME \
                noise.type=$NOISE \
                noise.scheduler_type=$SCHEDULER \
                sampling.num_steps=$STEPS \
                sampling.guid_scale=$GUID \
                sampling.batch_size=128 \
                sampling.num_batches=32
        done
    done
done

python -m scripts.metrics_images --run_name $RUN_NAME --dataset_name mnist \
    --to_evaluate_checkpoints 200 --diffusion_type $NOISE
