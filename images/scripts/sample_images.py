"""
Generate class-conditional samples (random classes) with classifier-free guidance.

Reads:  logs/<run>/checkpoints/checkpoint_<k>/ema_model.safetensors
Writes: logs/Figures/<run>/samples/ckpt_<k>/guid_<g>_num_steps_<n>_final_samples_<N>[_<scheduler>][_ancestral]/
          000000.png, 000001.png, ...  and  final_xs.pt (generated tokens)
        logs/Figures/<run>/diagnostics/  grid of the first samples
Number of samples: sampling.num_batches * sampling.batch_size.
"""
import os
import hydra
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from matplotlib import pyplot as plt
from PIL import Image

from vision_model.dataset import DatacollatorImages
from vision_model.model.dit_model import DiTConfig
from discrete_flow_matching.sampling import sample
from discrete_flow_matching.noise_scheduler import noising_images
from discrete_flow_matching.utils import set_all_seeds
from scripts.image_utils import tokens_to_rgb, tokens_to_display, load_dit_checkpoint


def save_samples(final_xs, vocab_size, save_dir, grid_path):
    """Save every sample as a PNG, the tokens as final_xs.pt, and a grid of the first samples."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(final_xs, f"{save_dir}/final_xs.pt")
    for i, x in enumerate(final_xs):
        Image.fromarray(tokens_to_rgb(x, vocab_size)).save(os.path.join(save_dir, f"{i:06d}.png"))

    grid_size = 8 if len(final_xs) >= 64 else 4
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    for i, ax in enumerate(axes.flatten()):
        ax.axis("off")
        if i < len(final_xs):
            ax.imshow(tokens_to_display(final_xs[i], vocab_size))
    plt.tight_layout()
    os.makedirs(os.path.dirname(grid_path), exist_ok=True)
    plt.savefig(grid_path.format(grid_size=grid_size), bbox_inches="tight", pad_inches=0.1)
    plt.close()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    set_all_seeds(1)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name         = cfg["sampling"]["run_name"]
    checkpoints      = cfg["sampling"]["to_evaluate_checkpoints"]
    num_steps        = cfg["sampling"]["num_steps"]
    guid_scale       = cfg["sampling"]["guid_scale"]
    num_batches      = cfg["sampling"]["num_batches"]
    use_ancestral    = cfg["sampling"]["use_ancestral_sampling"]
    batch_size       = cfg["sampling"]["batch_size"]
    diffusion_type   = cfg["noise"]["type"]
    scheduler_type   = cfg["noise"]["scheduler_type"]   # noise schedule used for sampling
    polynomial_power = cfg["noise"]["polynomial_power"]

    vocab_size  = cfg["model"]["vocab_size"] + (1 if diffusion_type == "masking" else 0)  # + MASK token
    num_classes = cfg["model"].get("num_classes", 10)                                     # index num_classes = unconditional
    print(f"{run_name}: {diffusion_type}, {scheduler_type} schedule, {num_steps} steps, guid_scale={guid_scale}, "
          f"{num_batches * batch_size} samples{' (ancestral sampler)' if use_ancestral else ''}")

    model_config = DiTConfig(
        hidden_size=cfg["model"]["hidden_size"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        frequency_embedding_size=cfg["model"]["frequency_embedding_size"],
        input_size=cfg["model"]["input_size"],
        in_channels=cfg["model"]["in_channels"],
        vocab_size=vocab_size,
    )

    # A data batch is only used for the image shape and to draw the fully noised starting point x_in.
    collator = DatacollatorImages(
        return_labels=True,
        diffusion_type=diffusion_type,
        scheduler_type=scheduler_type,
        polynomial_power=polynomial_power,
        t_max_train=cfg["trainer"]["t_max_train"],
        t_min_train=cfg["trainer"]["t_min_train"],
        vocab_size=vocab_size,
    )
    dataloader = DataLoader(load_from_disk(cfg["dataset"]["path"])["train"],
                            batch_size=batch_size, collate_fn=collator, shuffle=False)

    for ckpt in checkpoints:
        model = load_dit_checkpoint(run_name, ckpt, model_config, device)

        _, x_t_batch, _, y_batch = next(iter(dataloader))
        y_full = torch.where(y_batch == -100, x_t_batch, y_batch).to(device)
        t_start = torch.zeros(batch_size, device=device)

        final_xs = []
        for _ in tqdm(range(num_batches), desc=f"checkpoint {ckpt}"):
            c = torch.randint(0, num_classes, (batch_size,), device=device)
            c_uncond = torch.full((batch_size,), num_classes, dtype=torch.long, device=device)
            x_in, _ = noising_images(y_full, t_start, vocab_size, diffusion_type=diffusion_type,
                                     scheduler_type=scheduler_type, polynomial_power=polynomial_power)
            x, *_ = sample(model, c, c_uncond=c_uncond, guid_scale=guid_scale, vocab_size=vocab_size,
                           num_steps=num_steps, batch_size=batch_size, x_in=x_in,
                           diffusion_type=diffusion_type, scheduler_type=scheduler_type,
                           polynomial_power=polynomial_power, use_ancestral_sampling=use_ancestral,
                           return_trajectory=False)
            final_xs.append(x.cpu())
        final_xs = torch.cat(final_xs, dim=0)

        suffix = ("" if scheduler_type == "linear" else f"_{scheduler_type}") + ("_ancestral" if use_ancestral else "")
        setting = f"guid_{guid_scale}_num_steps_{num_steps}"
        save_samples(
            final_xs, vocab_size,
            save_dir=f"logs/Figures/{run_name}/samples/ckpt_{ckpt}/{setting}_final_samples_{len(final_xs)}{suffix}",
            grid_path=f"logs/Figures/{run_name}/diagnostics/ckpt_{ckpt}_generation_{setting}"
                      f"_sample_grid_{{grid_size}}x{{grid_size}}{suffix}.png",
        )


if __name__ == "__main__":
    main()
