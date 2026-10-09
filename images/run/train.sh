#!/bin/bash
# Train one of the four MNIST models of the paper: DiT-XS, 200k steps, effective batch size 512, EMA.
#
#   bash run/train.sh
#
# Extra Hydra overrides can be appended, e.g. bash run/train.sh trainer.max_steps=1000

NOISE=uniform                             # uniform | masking
RUN_NAME=MNIST_dit_xs_bs512_uniform_ema   # MNIST_dit_xs_bs512_{uniform,masking}_ema[_uncond]
LABEL_DROPOUT=0.15                        # 0.15: class-conditional model, 1.0: unconditional model (_uncond)

accelerate launch --config_file=conf/accelerate_ddp.yaml --num_processes=1 \
    scripts/train.py \
    hydra.run.dir=logs/$RUN_NAME \
    trainer.dir=logs/$RUN_NAME \
    wandb.dir=logs/$RUN_NAME \
    wandb.name=$RUN_NAME \
    wandb.project=mnist_dit \
    noise.type=$NOISE \
    trainer.label_dropout_prob=$LABEL_DROPOUT \
    trainer.max_steps=200000 \
    trainer.save_steps=1000 \
    trainer.use_ema=true \
    trainer.per_device_batch_size=128 \
    trainer.gradient_accumulation_steps=4 \
    "$@"
