"""
Generate samples that avoid one class with the NOT gate (sample_NOT), or with the static
negative-CFG baseline (sample_NOT_baseline) when sampling.not_baseline=true.

The NOT gate sweeps all combinations of sampling.not_temps (posterior temperature tau) and
sampling.not_initial_guid_scales; the baseline sweeps sampling.not_baseline_lams. The
unconditional branch uses the separately trained model sampling.uncond_run_name.

Reads:  logs/<run>/checkpoints/checkpoint_<k>/ema_model.safetensors
Writes: logs/Figures/<run>/not_gate/ckpt_<k>/<one folder per setting>/
          000000.png, 000001.png, ...
          post_probs.pt, guid_scales.pt   [T, N] posterior of the avoided class and guidance scale
                                          over time (NOT gate only)
Number of samples per setting: sampling.num_batches * sampling.batch_size.
"""

import os
import hydra
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from vision_model.dataset import DatacollatorImages
from vision_model.model.dit_model import DiTConfig
from discrete_flow_matching.sampling import sample_NOT, sample_NOT_baseline
from discrete_flow_matching.noise_scheduler import noising_images
from discrete_flow_matching.utils import set_all_seeds
from scripts.image_utils import tokens_to_rgb, load_dit_checkpoint


SEED_BASE = 1  # batch b of every setting uses seed SEED_BASE + b


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    set_all_seeds(1)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name                = cfg_dict["sampling"]["run_name"]
    uncond_run_name         = cfg_dict["sampling"].get("uncond_run_name", None)
    to_evaluate_checkpoints = cfg_dict["sampling"]["to_evaluate_checkpoints"]
    diffusion_type          = cfg_dict["noise"]["type"]
    scheduler_type          = cfg_dict["noise"]["scheduler_type"]
    polynomial_power        = cfg_dict["noise"]["polynomial_power"]
    t_max_train             = cfg_dict["trainer"]["t_max_train"]
    t_min_train             = cfg_dict["trainer"]["t_min_train"]
    num_steps                    = cfg_dict["sampling"]["num_steps"]
    num_batches                  = cfg_dict["sampling"]["num_batches"]
    inference_batch_size         = cfg_dict["sampling"]["batch_size"]
    use_ancestral_sampling       = cfg_dict["sampling"]["use_ancestral_sampling"]
    use_ancestral_sampling_probs = cfg_dict["sampling"]["use_ancestral_sampling_transition_likelihoods"]
    cache_dir                    = cfg_dict["dataset"]["path"]

    not_class      = int(cfg_dict["sampling"].get("not_class", 0))
    guid_scale_max = float(cfg_dict["sampling"].get("not_guid_scale_max", 5.0))
    temperatures         = [float(v) for v in cfg_dict["sampling"]["not_temps"]]
    initial_guid_scales  = [float(v) for v in cfg_dict["sampling"]["not_initial_guid_scales"]]
    not_baseline         = bool(cfg_dict["sampling"].get("not_baseline", False))
    baseline_lams        = [float(v) for v in cfg_dict["sampling"].get("not_baseline_lams", [0.5, 1.0, 2.0, 3.0])]

    if diffusion_type == 'masking':
        cfg_dict["model"]["vocab_size"] += 1
    vocab_size  = cfg_dict["model"]["vocab_size"]
    num_classes = cfg_dict["model"].get("num_classes", 10)

    num_samples = num_batches * inference_batch_size
    print(f"{run_name}: NOT {not_class}, {diffusion_type}, {scheduler_type} schedule, {num_steps} steps, "
          f"{num_samples} samples per setting, unconditional model: {uncond_run_name or run_name}")
    print(f"  baseline lams={baseline_lams}" if not_baseline
          else f"  temps={temperatures}  initial_guid_scales={initial_guid_scales}  guid_scale_max={guid_scale_max}")

    model_config = DiTConfig(
        hidden_size=cfg_dict["model"]["hidden_size"],
        depth=cfg_dict["model"]["depth"],
        num_heads=cfg_dict["model"]["num_heads"],
        frequency_embedding_size=cfg_dict["model"]["frequency_embedding_size"],
        input_size=cfg_dict["model"]["input_size"],
        in_channels=cfg_dict["model"]["in_channels"],
        vocab_size=vocab_size,
    )

    # A data batch is only used for the image shape and to draw the fully noised starting point x_in.
    ds = load_from_disk(cache_dir)
    collator = DatacollatorImages(
        return_labels=True,
        diffusion_type=diffusion_type,
        scheduler_type=scheduler_type,
        polynomial_power=polynomial_power,
        t_max_train=t_max_train,
        t_min_train=t_min_train,
        vocab_size=vocab_size,
    )
    dataloader = DataLoader(
        ds["test"],
        batch_size=inference_batch_size,
        collate_fn=collator,
        shuffle=False,
    )

    out_base_root = f"logs/Figures/{run_name}/not_gate"

    for checkpoint_number in to_evaluate_checkpoints:
        print(f"\n{'='*40}\nCheckpoint {checkpoint_number}\n{'='*40}")

        cond_model = load_dit_checkpoint(run_name, checkpoint_number, model_config, device)
        uncond_model = cond_model
        if uncond_run_name is not None:
            uncond_model = load_dit_checkpoint(uncond_run_name, checkpoint_number, model_config, device)

        c_batch, x_t_batch, t_batch, y_batch = next(iter(dataloader))
        y_full = torch.where(y_batch == -100, x_t_batch, y_batch).to(device)
        B_data, C, H, W = y_full.shape

        out_base = f"{out_base_root}/ckpt_{checkpoint_number}"
        os.makedirs(out_base, exist_ok=True)

        # (temp, initial_guid_scale) pairs for the NOT gate, lam values for the baseline
        sweep_settings = baseline_lams if not_baseline else [
            (temp, igs) for temp in temperatures for igs in initial_guid_scales
        ]

        for setting in sweep_settings:
            lik_tag = "ancestral" if use_ancestral_sampling_probs else "ode"
            denoise_suffix = "_denoise_ancestral" if use_ancestral_sampling else ""
            # .4f, not .2f: metrics_not_gate.py parses the settings back out of the folder name,
            # and at 2 decimals tau=0.025 reads back as 0.03 and tau=0.005 collides with 0.01.
            if not_baseline:
                lam = setting
                fname = (f"not_baseline_{not_class}_lam_{lam:.4f}"
                         f"_sched_{scheduler_type}_n_{num_samples}{denoise_suffix}")
            else:
                temp, initial_guid_scale = setting
                fname = (f"not_{not_class}_temp_{temp:.4f}_initgs_{initial_guid_scale:.4f}"
                         f"_gsmax_{guid_scale_max:.2f}_sampler_{lik_tag}"
                         f"_sched_{scheduler_type}_n_{num_samples}{denoise_suffix}")
            save_dir = os.path.join(out_base, fname)
            os.makedirs(save_dir, exist_ok=True)
            print(f"\n── {fname}")

            all_imgs, all_post, all_guid = [], [], []
            for batch_idx in tqdm(range(num_batches), desc="batches"):
                # Same seed for batch b of every setting: all settings start from the same
                # noise, so differences between settings are not due to different random draws.
                seed = SEED_BASE + batch_idx
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                c        = torch.full((inference_batch_size,), not_class,   dtype=torch.long, device=device)
                c_uncond = torch.full((inference_batch_size,), num_classes, dtype=torch.long, device=device)

                t_start = torch.zeros(inference_batch_size, device=device)
                x_in, _ = noising_images(
                    y_full[:inference_batch_size], t_start, vocab_size,
                    diffusion_type=diffusion_type,
                    scheduler_type=scheduler_type,
                    polynomial_power=polynomial_power,
                )

                if not_baseline:
                    final_x, _ = sample_NOT_baseline(
                        cond_model, c, c_uncond,
                        lam_constant=lam,
                        x_in=x_in,
                        diffusion_type=diffusion_type,
                        scheduler_type=scheduler_type,
                        polynomial_power=polynomial_power,
                        use_ancestral_sampling=use_ancestral_sampling,
                        num_steps=num_steps,
                        vocab_size=vocab_size,
                        batch_size=inference_batch_size, C=C, H=H, W=W,
                        return_trajectory=True,
                        uncond_model=uncond_model,
                    )
                    all_imgs.append(final_x.cpu())
                else:
                    final_x, _, post_probs, guid_scales = sample_NOT(
                        cond_model, c, c_uncond,
                        guid_scale_max=guid_scale_max,
                        initial_guid_scale=initial_guid_scale,
                        x_in=x_in,
                        diffusion_type=diffusion_type,
                        scheduler_type=scheduler_type,
                        polynomial_power=polynomial_power,
                        use_ancestral_sampling=use_ancestral_sampling,
                        use_ancestral_sampling_probs=use_ancestral_sampling_probs,
                        num_steps=num_steps,
                        vocab_size=vocab_size,
                        batch_size=inference_batch_size, C=C, H=H, W=W,
                        temp=temp,
                        return_trajectory=True,
                        uncond_model=uncond_model,
                    )
                    all_imgs.append(final_x.cpu())
                    all_post.append(post_probs.cpu())     # [T, bs]
                    all_guid.append(guid_scales.cpu())     # [T, bs]

            all_imgs = torch.cat(all_imgs, dim=0)   # [N, C, H, W]

            if not not_baseline:
                post_probs_all  = torch.cat(all_post, dim=1)   # [T, N]
                guid_scales_all = torch.cat(all_guid, dim=1)   # [T, N]
                torch.save(post_probs_all, os.path.join(save_dir, "post_probs.pt"))
                torch.save(guid_scales_all, os.path.join(save_dir, "guid_scales.pt"))

            for i, img in enumerate(all_imgs):
                img_rgb = tokens_to_rgb(img, vocab_size)
                Image.fromarray(img_rgb).save(os.path.join(save_dir, f"{i:06d}.png"))

            print(f"   → {len(all_imgs)} images saved to {save_dir}")


if __name__ == "__main__":
    main()
