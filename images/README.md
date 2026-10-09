# Image experiments

Code for the MNIST experiments of
*Online Class Posteriors for Interpretable and Controllable Discrete Diffusion*.

A discrete diffusion model on raw pixel values (vocabulary of 256 tokens) is trained with either
**masking** or **uniform** noise, using a DiT backbone with AdaLN-Zero class conditioning.
On top of it we implement:

- the **online class posterior**, tracked during sampling (`sampler_with_posterior`),
- the **NOT gate** (dynamic negative guidance driven by the posterior, `sample_NOT`) and its static
  negative-CFG baseline (`sample_NOT_baseline`).

The OR gate (`sample_OR`) is implemented alongside the NOT gate but not evaluated on images in the paper.

The posterior sampler lives in [`sampling_standard.py`](src/discrete_flow_matching/sampling/sampling_standard.py)
and the NOT / OR gates in [`sampling_logical_operators.py`](src/discrete_flow_matching/sampling/sampling_logical_operators.py)
(both in `src/discrete_flow_matching/sampling/`).

**Time convention:** `t=0` is pure noise, `t=1` is clean data; sampling integrates from 0 to 1.

## Setup

```bash
# In a fresh environment (Python >= 3.10; we used 3.13). Install torch/torchvision for your CUDA version first, e.g.
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .                     # makes src/vision_model and src/discrete_flow_matching importable

python -m scripts.prepare_data       # downloads MNIST into data/ and exports the FID reference PNGs
```

Use a dedicated environment: if another project installs packages named `vision_model` or
`discrete_flow_matching` in the same environment, Python may import those instead of the ones in `src/`.

All commands below are run from this folder. Logging to Weights & Biases is off by default;
`export WANDB_MODE=online` to enable it.

## Pretrained checkpoints

EMA weights of the four models used in the paper (DiT-XS, 7.6M parameters for 200k steps):

| Run name | Noise | Conditioning |
|---|---|---|
| `MNIST_dit_xs_bs512_uniform_ema` | uniform | class-conditional |
| `MNIST_dit_xs_bs512_uniform_ema_uncond` | uniform | unconditional |
| `MNIST_dit_xs_bs512_masking_ema` | masking | class-conditional |
| `MNIST_dit_xs_bs512_masking_ema_uncond` | masking | unconditional |

Download `mnist_ema_checkpoints.tar.gz` from [OSF](https://osf.io/veqbg/overview?view_only=a03e63439fd24ad0b27105a694ecc0e8) and extract it here:

```bash
tar xzf mnist_ema_checkpoints.tar.gz   # creates logs/<run_name>/checkpoints/checkpoint_200/ema_model.safetensors
```

Checkpoints are numbered by save, one every 1000 training steps (`trainer.save_steps` in `run/train.sh`),
so `checkpoint_200` is the model after 200k steps.

The class posterior and the gates combine a class-conditional model with the *separately trained*
unconditional model of the same noise type (`sampling.uncond_run_name`; we found this to work better
than the null token of label dropout), so both are needed.

## Reproducing the results

Each script in `run/` starts with a settings block (noise type, run name, ...) to adapt; the commented
lines at the bottom of the gate scripts list the grids used in the paper. Samples are written to
`logs/Figures/<run_name>/`, metrics (CSV) to `logs/Results/<run_name>/`.

| Experiment | Command |
|---|---|
| Training (4 models: `NOISE` uniform/masking, `LABEL_DROPOUT` 0.15 conditional / 1.0 unconditional) | `bash run/train.sh` |
| Class posterior vs. ViT classifier, example posterior trajectories | `bash run/posterior.sh` |
| NOT gate: τ × initial-guidance grid, Pareto points | `bash run/not_gate.sh` |
| NOT gate: static negative-CFG baseline | `bash run/not_baseline.sh` |
| Appendix ablations: FID, precision/recall, C2ST vs. guidance scale, #steps and noise schedule | `bash run/ablations.sh` |

Unless stated otherwise, sampling uses 128 steps of a cosine schedule `κ(t) = 1 − cos(πt/2)`.
`run/train.sh` forwards extra Hydra overrides, e.g. `bash run/train.sh trainer.max_steps=1000`.

**Configuration.** `conf/config.yaml` composes the Hydra configs. The noise process is shared by training and
sampling: `noise.type` (`uniform` | `masking`), `noise.scheduler_type` (the κ(t) used to noise the data
during training, and the time grid during sampling) and `noise.polynomial_power` (only for the polynomial schedules). The training objective is in `conf/trainer/dfm.yaml`,
the training loop in `conf/trainer/base.yaml`, the sampling settings in `conf/sampling/`.

**Other datasets.** The model and pipeline handle RGB input (`in_channels`, `input_size` in
`conf/model/`). Adding a dataset requires a config in `conf/dataset/`, an entry in
`scripts/prepare_data.py`, and adapting the settings block at the top of the `run/` scripts.

## Layout

```
src/
  vision_model/                 DiT model, data collator, loss, training loop (Accelerate, EMA)
  discrete_flow_matching/
    noise_scheduler/            κ(t) schedules, forward noising (masking / uniform)
    sampling/                   sampler, CFG, posterior tracking (sampling_standard.py),
                                NOT / OR gates and baselines (sampling_logical_operators.py)
    metrics/                    FID, k-NN precision/recall and C2ST on Inception features
scripts/                        entry points called by run/*.sh
conf/                           Hydra configs (noise, model, dataset, trainer, sampling)
run/                            one script per experiment family
```

## Acknowledgements

The training infrastructure is adapted from [AMPLIFY](https://github.com/chandar-lab/AMPLIFY) (MIT);
the DiT model from [DiT](https://github.com/facebookresearch/DiT) (Peebles & Xie, 2023; CC BY-NC 4.0);
the k-NN precision and recall from [NVIDIA's implementation](https://github.com/kynkaat/improved-precision-and-recall-metric)
(Kynkäänniemi et al., 2019; CC BY-NC 4.0). See `LICENSE` for which files fall under which license.
