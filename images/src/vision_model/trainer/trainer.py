import os
import re
import sys
import torch
import signal
import shutil
from contextlib import nullcontext
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
import safetensors.torch as safetensors_torch

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from datasets import load_from_disk
from torch.utils.data import DataLoader
from ..dataset import DatacollatorImages
from ..model.dit_model import DiT, DiTConfig
from ..metric import Metrics
from ..loss import get_loss, get_loss_weight
from ..scheduler import get_scheduler
from ..optimizer import get_optimizer
from discrete_flow_matching.noise_scheduler import get_scheduler_images

# Accelerate's checkpoint loading needs torch.load(weights_only=False) on recent PyTorch versions.
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load


def compute_loss(model, batch, loss_fn, cfg):
    """Forward pass with CFG label dropout. Returns logits, per-token loss [B, L], its weighted version, and the
    weighted loss averaged over the active (non -100) tokens."""
    c, x_t, t, y = batch
    loss_weight = get_loss_weight(t, max_weight_loss=cfg.trainer.max_weight_loss,
                                  scheduler_type=cfg.noise.scheduler_type, polynomial_power=cfg.noise.polynomial_power)
    if cfg.trainer.label_dropout_prob > 0:
        force_drop_ids = torch.rand(c.shape[0], device=c.device) < cfg.trainer.label_dropout_prob
    else:
        force_drop_ids = None
    # The model is conditioned on the noise level kappa(t), as in the samplers.
    # In this work we train with a linear scheduler, such that kappa(t) = t.
    kappa_t, _ = get_scheduler_images(t, scheduler_type=cfg.noise.scheduler_type, polynomial_power=cfg.noise.polynomial_power)
    logits = model(x_t, kappa_t, c, force_drop_ids=force_drop_ids)

    per_token_loss = loss_fn(logits.reshape(-1, cfg.model.vocab_size), y.reshape(-1)).reshape(y.shape[0], -1)
    if cfg.trainer.use_loss_weighting:
        weighted_per_token_loss = loss_weight.reshape(-1, 1) * per_token_loss
    else:
        weighted_per_token_loss = per_token_loss
    loss = weighted_per_token_loss.sum() / (y != -100).sum().clamp(min=1)
    return logits, per_token_loss, weighted_per_token_loss, loss


def evaluate(model, dataloader, loss_fn, cfg):
    """Validation loss and accuracy over the whole dataloader, plus the (weighted) loss per bin of t.

    Returns: number of predicted tokens, summed loss, number of correct predictions, dict of per-bin losses.
    """
    was_training = model.training
    model.eval()
    num_bins = cfg.trainer.num_bins
    device = next(model.parameters()).device
    sum_val_loss, num_val_correct, num_val_pred = 0, 0, 0
    bin_sums, bin_sums_weighted, bin_counts = (torch.zeros(num_bins).to(device) for _ in range(3))

    with torch.no_grad():
        for c, x_t, t, y in dataloader:
            logits, per_token_loss, weighted_per_token_loss, val_loss = compute_loss(model, (c, x_t, t, y), loss_fn, cfg)

            num_active = torch.sum(y != -100).item()
            num_val_pred += num_active
            sum_val_loss += val_loss.item() * num_active
            num_val_correct += torch.sum(torch.argmax(logits, dim=-1) == y).item()

            # Per-sample average loss, accumulated in the bin of its noise level t
            num_active_per_sample = (y != -100).sum(dim=[1, 2, 3]).float().clamp(min=1)  # [B]
            per_sample_avg_loss = per_token_loss.sum(dim=1) / num_active_per_sample
            weighted_per_sample_avg_loss = weighted_per_token_loss.sum(dim=1) / num_active_per_sample
            bin_indices = (t * num_bins).clamp(0, num_bins - 1).long()
            bin_sums.index_add_(0, bin_indices, per_sample_avg_loss)
            bin_sums_weighted.index_add_(0, bin_indices, weighted_per_sample_avg_loss)
            bin_counts.index_add_(0, bin_indices, torch.ones_like(per_sample_avg_loss))

    bin_averages = {**{f"bin_{i}": (bin_sums[i] / bin_counts[i]).item() if bin_counts[i] > 0 else 0 for i in range(num_bins)},
                    **{f"bin_weighted_{i}": (bin_sums_weighted[i] / bin_counts[i]).item() if bin_counts[i] > 0 else 0 for i in range(num_bins)}}

    model.train(was_training)
    return num_val_pred, sum_val_loss, num_val_correct, bin_averages


def latest_checkpoint(chk_dir):
    """Highest checkpoint index in chk_dir (same pattern as accelerator.load_state())."""
    return max(int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)[0]) for folder in os.listdir(chk_dir))


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def trainer(cfg: DictConfig) -> None:
    """Entrypoint for training a model with the given Hydra configuration."""
    it = 0
    chk_dir = os.path.join(cfg.trainer.dir, "checkpoints")
    # Delete the folder if resume is disabled and the folder exists
    if cfg.trainer.resume is False:
        shutil.rmtree(chk_dir, ignore_errors=True)
    elif os.path.exists(chk_dir):
        it = latest_checkpoint(chk_dir)
        # Remove empty checkpoint folders
        while len(os.listdir(os.path.join(chk_dir, f"checkpoint_{it}"))) == 0:
            shutil.rmtree(os.path.join(chk_dir, f"checkpoint_{it}"), ignore_errors=True)
            it -= 1

    # Accelerator object
    project_config = ProjectConfiguration(
        cfg.trainer.dir,
        automatic_checkpoint_naming=True,
        total_limit=cfg.trainer.max_checkpoints,
        iteration=it + 1,
    )
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb",
        project_config=project_config,
    )

    # Initialise the wandb run and pass wandb parameters
    os.makedirs(cfg.wandb.dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg.wandb.name,
                "entity": cfg.wandb.entity,
                "config": OmegaConf.to_container(cfg)
                | {"distributed_type": accelerator.distributed_type}
                | {"mixed_precision": accelerator.mixed_precision},
                "tags": cfg.wandb.tags,
                "dir": cfg.wandb.dir,
                "mode": cfg.wandb.mode,
                "anonymous": "allow",
                "resume": cfg.trainer.resume,
            }
        },
    )

    set_seed(cfg.seed)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.trainer.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.trainer.tf32)

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)

    # Model, optimizer, and learning rate scheduler. Masking adds an extra MASK token to the vocabulary.
    if cfg.noise.type == "masking":
        cfg.model.vocab_size += 1
    model = DiT(DiTConfig(**cfg.model))
    optimizer = get_optimizer(model, **cfg.optimizer)
    scheduler = get_scheduler(optimizer, **cfg.scheduler)
    accelerator.log({"model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    # Dataloaders; the collator noises the images, with the same settings for training and validation
    ds = load_from_disk(cfg.dataset.path)
    collator = DatacollatorImages(
        return_labels=True,
        diffusion_type=cfg.noise.type,
        scheduler_type=cfg.noise.scheduler_type,
        polynomial_power=cfg.noise.polynomial_power,
        t_max_train=cfg.trainer.t_max_train,
        t_min_train=cfg.trainer.t_min_train,
        vocab_size=cfg.model.vocab_size,
    )
    train_dataloader = DataLoader(ds["train"], batch_size=cfg.trainer.per_device_batch_size, collate_fn=collator, shuffle=True)
    eval_dataloader = DataLoader(ds["test"], batch_size=cfg.trainer.per_device_batch_size, collate_fn=collator, shuffle=True)

    # Accelerate
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)
    eval_dataloader = accelerator.prepare(eval_dataloader)

    # EMA: full model copy, kept in eval mode, no gradients
    if cfg.trainer.use_ema:
        ema_model = deepcopy(accelerator.unwrap_model(model)).to(accelerator.device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    loss_fn = get_loss(label_smoothing=cfg.trainer.label_smoothing)

    # Resume from the latest checkpoint
    skipped_train_dataloader = None
    if cfg.trainer.resume and os.path.exists(chk_dir):
        accelerator.load_state()
        skipped_train_dataloader = accelerator.skip_first_batches(train_dataloader, metrics["num_batches_in_epoch"])
        if cfg.trainer.use_ema:
            ema_resume_path = os.path.join(chk_dir, f"checkpoint_{it}", "ema_model.safetensors")
            if os.path.exists(ema_resume_path):
                ema_model.load_state_dict(safetensors_torch.load_file(ema_resume_path, device="cpu"))
                ema_model.to(accelerator.device)

    def save_checkpoint():
        # Accelerate saves model, optimizer, scheduler, RNG states and metrics; the EMA weights are added to the same folder
        accelerator.save_state()
        if cfg.trainer.use_ema and accelerator.is_main_process:
            ema_path = os.path.join(chk_dir, f"checkpoint_{latest_checkpoint(chk_dir)}", "ema_model.safetensors")
            safetensors_torch.save_file(ema_model.state_dict(), ema_path)

    # Save the model when receiving the signal SIGTERM
    def handler(signum, frame):
        print(f"Signal {signum} received on rank {accelerator.process_index}, checkpointing...")
        save_checkpoint()
        accelerator.wait_for_everyone()
        print(f"Done on rank {accelerator.process_index}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)

    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=metrics["num_steps"],
        total=cfg.trainer.max_steps,
        disable=(cfg.trainer.disable_tqdm or not accelerator.is_main_process),
    )

    while cfg.trainer.max_steps > metrics["num_steps"]:
        # Use skipped_train_dataloader the first epoch after resuming
        dataloader = train_dataloader if skipped_train_dataloader is None else skipped_train_dataloader

        for c, x_t, t, y in dataloader:
            metrics["local_num_batches"] += 1
            # Gradient accumulation: gradients are only synchronised across processes on the last micro-batch
            # of each cycle, which is also the one that takes the optimizer step
            is_step = metrics["local_num_batches"] % cfg.trainer.gradient_accumulation_steps == 0

            with nullcontext() if is_step else accelerator.no_sync(model):
                logits, _, _, train_loss = compute_loss(model, (c, x_t, t, y), loss_fn, cfg)

                if is_step:
                    pbar.update(1)
                    metrics["num_steps"] += 1
                num_active = torch.sum(y != -100).item()
                metrics["num_batches_in_epoch"] += 1
                metrics["local_num_samples"] += x_t.shape[0]
                metrics["local_num_tokens"] += x_t.numel()
                metrics["local_num_train_pred"] += num_active
                metrics["local_sum_train_loss"] += train_loss.item() * num_active
                metrics["local_num_train_correct"] += torch.sum(torch.argmax(logits, dim=-1) == y).item()

                accelerator.backward(train_loss)

            if not is_step:
                continue

            # Evaluate the model (the EMA model if enabled)
            if metrics["num_steps"] % cfg.trainer.eval_steps == 0:
                num_val_pred, sum_val_loss, num_val_correct, bin_avgs = evaluate(
                    ema_model if cfg.trainer.use_ema else model, eval_dataloader, loss_fn, cfg
                )
                metrics["local_test_sum_val_loss"] = sum_val_loss
                metrics["local_test_num_val_correct"] = num_val_correct
                metrics["local_test_num_val_pred"] = num_val_pred
                for bin_name, avg_loss in bin_avgs.items():
                    metrics[f"local_test_bins/loss_t_{bin_name}"] = avg_loss

            # Log metrics (the gradient norm is the one before clipping)
            if metrics["num_steps"] % cfg.wandb.log_interval == 0:
                metrics["grad_norm"] = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                metrics["weight_norm"] = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
                metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
                metrics.log(accelerator, os.path.join(cfg.wandb.dir, "wandb", "metrics.json"))

            if cfg.trainer.gradient_clipping is not None and cfg.trainer.gradient_clipping > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)

            optimizer.step()
            scheduler.step()
            if cfg.trainer.use_ema:
                update_ema(ema_model, accelerator.unwrap_model(model), cfg.trainer.ema_decay)
            optimizer.zero_grad()

            if metrics["num_steps"] % cfg.trainer.save_steps == 0:
                save_checkpoint()

            if metrics["num_steps"] >= cfg.trainer.max_steps:
                break

        metrics["num_epochs"] += 1
        metrics["num_batches_in_epoch"] = 0

        # "Remove" the skipped dataloader once exhausted
        skipped_train_dataloader = None

    # Make sure that the wandb tracker finishes correctly and close the progress bar
    pbar.close()
    accelerator.end_training()
