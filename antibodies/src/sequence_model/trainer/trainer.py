import os
import re
import sys
import torch
import signal
import shutil
from contextlib import nullcontext
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from ..model import AMPLIFY, AMPLIFYConfig
from ..metric import Metrics
from ..loss import get_loss, get_loss_weight
from ..dataset import get_dataloader
from ..scheduler import get_scheduler
from ..optimizer import get_optimizer

# Accelerate's checkpoint loading needs torch.load(weights_only=False) on recent PyTorch
# versions; AMPLIFY was written against older ones, where this was the default.
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load


def compute_loss(model, batch, loss_fn, cfg, force_drop_ids=None):
    """Forward pass and loss. Returns logits, the per-token loss [B, L], its weighted version,
    and the weighted loss averaged over the active (non -100) tokens.

    force_drop_ids is drawn by the training loop and passed in, rather than sampled here as in
    the image experiments' counterpart: validation must report the fully-conditional loss, and
    model.eval() alone already makes LabelEmbedder skip its own dropout. Passing None (as
    evaluate does) therefore keeps the CFG label dropout out of the validation numbers.

    The time weighting uses the *training* objective's settings for both splits, as before:
    cfg.trainer.validation only configures the validation collator.
    """
    x, t, y, pad_mask, class_labels = batch
    loss_weight = get_loss_weight(t, max_weight_loss=cfg.trainer.max_weight_loss,
                                  scheduler_type=cfg.trainer.train.scheduler_type,
                                  polynomial_power=cfg.trainer.train.polynomial_power)
    logits = model(x, t, pad_mask, class_labels=class_labels, force_drop_ids=force_drop_ids).logits

    per_token_loss = loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), y.view(-1)).view(y.size(0), -1)
    if cfg.trainer.train.use_loss_weighting:
        weighted_per_token_loss = loss_weight.unsqueeze(1) * per_token_loss
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
    num_bins = cfg.trainer.train.num_bins
    device = next(model.parameters()).device
    sum_val_loss, num_val_correct, num_val_pred = 0, 0, 0
    bin_sums, bin_sums_weighted, bin_counts = (torch.zeros(num_bins).to(device) for _ in range(3))

    with torch.no_grad():
        for batch in dataloader:
            logits, per_token_loss, weighted_per_token_loss, val_loss = compute_loss(model, batch, loss_fn, cfg)
            _, t, y, _, _ = batch

            num_active = torch.sum(y != -100).item()
            num_val_pred += num_active
            sum_val_loss += val_loss.item() * num_active
            num_val_correct += torch.sum(torch.argmax(logits, dim=-1) == y).item()

            # Per-sample average loss, accumulated in the bin of its noise level t
            num_active_per_sample = (y != -100).sum(dim=1).float().clamp(min=1)  # [B]
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

    # Model, optimizer, and learning rate scheduler
    model = AMPLIFY(AMPLIFYConfig(**cfg.model, **cfg.tokenizer))
    optimizer = get_optimizer(model, **cfg.optimizer)
    scheduler = get_scheduler(optimizer, **cfg.scheduler)
    accelerator.log({"model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    # The pad_mask is additive and is added to the attention scores, so it has to match the
    # autocast dtype of the forward pass.
    dtype_pad_mask = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

    # Dataloaders; the collator tokenizes and noises the sequences
    train_dataloader = get_dataloader(
        **cfg.tokenizer,
        **cfg.dataset.train,
        **cfg.trainer.train,
        merge=True,
        return_labels=False,
        dtype=dtype_pad_mask,
    )
    eval_dataloaders = get_dataloader(
        **cfg.tokenizer,
        **cfg.dataset.validation,
        **cfg.trainer.validation,
        merge=False,
        return_labels=False,
        dtype=dtype_pad_mask,
    )

    # Accelerate
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)
    eval_dataloaders = {k: accelerator.prepare(v) for k, v in eval_dataloaders.items()}

    train_loss_fn = get_loss(label_smoothing=cfg.trainer.train.label_smoothing)
    val_loss_fn = get_loss(label_smoothing=cfg.trainer.validation.label_smoothing)

    # Resume from the latest checkpoint
    skipped_train_dataloader = None
    if cfg.trainer.resume and os.path.exists(chk_dir):
        accelerator.load_state()
        skipped_train_dataloader = accelerator.skip_first_batches(train_dataloader, metrics["num_batches_in_epoch"])

    # Save the model when receiving the signal SIGTERM
    def handler(signum, frame):
        print(f"Signal {signum} received on rank {accelerator.process_index}, checkpointing...")
        accelerator.save_state()
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

        for batch in dataloader:
            x, t, y, pad_mask, class_labels = batch
            # CFG label dropout: replace the label by the unconditional token for a random subset
            force_drop_ids = (torch.rand(class_labels.shape[0], device=class_labels.device)
                              < cfg.trainer.train.label_dropout_prob).long()

            metrics["local_num_batches"] += 1
            # Gradient accumulation: gradients are only synchronised across processes on the last
            # micro-batch of each cycle, which is also the one that takes the optimizer step
            is_step = metrics["local_num_batches"] % cfg.trainer.gradient_accumulation_steps == 0

            with nullcontext() if is_step else accelerator.no_sync(model):
                logits, _, _, train_loss = compute_loss(model, batch, train_loss_fn, cfg,
                                                        force_drop_ids=force_drop_ids)

                if is_step:
                    pbar.update(1)
                    metrics["num_steps"] += 1
                num_active = torch.sum(y != -100).item()
                metrics["num_batches_in_epoch"] += 1
                metrics["local_num_samples"] += x.shape[0]
                metrics["local_num_tokens"] += (pad_mask == 0).sum().item()
                metrics["local_num_train_pred"] += num_active
                metrics["local_sum_train_loss"] += train_loss.item() * num_active
                metrics["local_num_train_correct"] += torch.sum(torch.argmax(logits, dim=-1) == y).item()

                accelerator.backward(train_loss)

            if not is_step:
                continue

            # Evaluate the model
            if metrics["num_steps"] % cfg.trainer.eval_steps == 0:
                for k, v in eval_dataloaders.items():
                    num_val_pred, sum_val_loss, num_val_correct, bin_avgs = evaluate(model, v, val_loss_fn, cfg)
                    metrics[f"local_{k}_sum_val_loss"] = sum_val_loss
                    metrics[f"local_{k}_num_val_correct"] = num_val_correct
                    metrics[f"local_{k}_num_val_pred"] = num_val_pred
                    for bin_name, avg_loss in bin_avgs.items():
                        metrics[f"local_{k}_bins/loss_t_{bin_name}"] = avg_loss

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
            optimizer.zero_grad()

            if metrics["num_steps"] % cfg.trainer.save_steps == 0:
                accelerator.save_state()

            if metrics["num_steps"] >= cfg.trainer.max_steps:
                break

        metrics["num_epochs"] += 1
        metrics["num_batches_in_epoch"] = 0

        # "Remove" the skipped dataloader once exhausted
        skipped_train_dataloader = None

    # Make sure that the wandb tracker finishes correctly and close the progress bar
    pbar.close()
    accelerator.end_training()
