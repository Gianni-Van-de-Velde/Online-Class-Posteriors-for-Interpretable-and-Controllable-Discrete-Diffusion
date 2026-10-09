import math
import json
from torch import Tensor
from accelerate import Accelerator
from collections import defaultdict
import wandb


class Metrics(defaultdict):
    """
    Training bookkeeping: a dict of counters, checkpointed with the Accelerator so it survives resumes.

    The trainer accumulates per-process counters under "local_*" keys (samples, tokens, summed
    train/validation loss, correct predictions, validation loss per noise-level bin). `log` sums
    them over processes, turns them into loss / perplexity / accuracy, logs these to W&B and to a
    JSON file, and resets the local counters.
    """

    def __init__(self):
        super().__init__(int)

    def state_dict(self):
        return dict(self)

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            self[k] = v

    def log(self, accelerator: Accelerator, json_path: str):
        # Sum the counters over all processes (only meaningful for the local_* counters)
        metrics_agg = Tensor(list(self.values())).to(accelerator.device, non_blocking=True)
        metrics_agg = accelerator.reduce(metrics_agg, reduction="sum").detach().cpu().numpy()
        metrics_agg = {k: v for k, v in zip(self.keys(), metrics_agg)}

        # Update the global counters
        self["num_samples"] = self["num_samples"] + metrics_agg["local_num_samples"]
        self["num_tokens"] = self["num_tokens"] + metrics_agg["local_num_tokens"]
        self["num_masked_tokens"] = self["num_masked_tokens"] + metrics_agg["local_num_train_pred"]

        metrics_log = dict()
        metrics_log["num_epochs"] = self["num_epochs"]
        metrics_log["num_steps"] = self["num_steps"]
        metrics_log["grad_norm_before_clip"] = self["grad_norm"]
        metrics_log["weight_norm"] = self["weight_norm"]
        metrics_log["learning_rate"] = self["learning_rate"]
        metrics_log["num_samples"] = self["num_samples"]
        metrics_log["num_tokens"] = self["num_tokens"]
        metrics_log["num_masked_tokens"] = self["num_masked_tokens"]
        if metrics_agg["local_num_train_pred"] > 0:
            metrics_log["train_loss"] = metrics_agg["local_sum_train_loss"] / metrics_agg["local_num_train_pred"]
            metrics_log["train_perplexity"] = math.exp(
                metrics_agg["local_sum_train_loss"] / metrics_agg["local_num_train_pred"]
            )
            metrics_log["train_accuracy"] = metrics_agg["local_num_train_correct"] / metrics_agg["local_num_train_pred"]
        for eval_set in set(k.split("_")[1] for k in self.keys() if "_val_" in k):
            metrics_log[f"{eval_set}_val_loss"] = (
                metrics_agg[f"local_{eval_set}_sum_val_loss"] / metrics_agg[f"local_{eval_set}_num_val_pred"]
            )
            metrics_log[f"{eval_set}_val_perplexity"] = math.exp(
                metrics_agg[f"local_{eval_set}_sum_val_loss"] / metrics_agg[f"local_{eval_set}_num_val_pred"]
            )
            metrics_log[f"{eval_set}_val_accuracy"] = (
                metrics_agg[f"local_{eval_set}_num_val_correct"] / metrics_agg[f"local_{eval_set}_num_val_pred"]
            )

            # Validation loss per noise-level bin, averaged over processes
            bin_keys = [k for k in metrics_agg.keys() if f"local_{eval_set}_bins/loss_t_bin" in k]
            current_eval_set_bins = {}
            for bk in bin_keys:
                log_key = bk.replace("local_", "")
                val = metrics_agg[bk] / accelerator.num_processes
                metrics_log[log_key] = val
                current_eval_set_bins[log_key] = val

            # Single W&B chart of the loss against the noise-level bin
            if accelerator.is_main_process and current_eval_set_bins:
                sorted_items = sorted(current_eval_set_bins.items(), key=lambda x: int(x[0].split('_')[-1]))
                xs = [int(k.split('_')[-1]) for k, v in sorted_items]
                ys = [v for k, v in sorted_items]
                metrics_log[f"{eval_set}/loss_by_noise_level"] = wandb.plot.line_series(
                    xs=xs,
                    ys=[ys],
                    keys=["CrossEntropy"],
                    title=f"Weighted loss per noise level - {eval_set}",
                    xname="Noise bin (0 = noisiest)",
                )

        accelerator.log(metrics_log)

        # Reset the local counters
        for k in metrics_agg.keys():
            if "local" in k:
                self.pop(k)

        # Append to the JSON log, which holds a single list of dicts
        if accelerator.is_main_process:
            metrics_log = {k: str(v) for k, v in metrics_log.items()}
            with open(json_path, "ab+") as f:
                f.seek(0, 2)
                if f.tell() == 0:
                    f.write(json.dumps([metrics_log]).encode())
                else:
                    f.seek(-1, 2)
                    f.truncate()  # remove the closing "]"
                    f.write(" , ".encode())
                    f.write(json.dumps(metrics_log).encode())
                    f.write("]".encode())
