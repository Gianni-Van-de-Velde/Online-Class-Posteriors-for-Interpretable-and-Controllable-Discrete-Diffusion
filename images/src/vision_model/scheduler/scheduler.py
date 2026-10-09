import math
import torch
from functools import partial
from torch.optim.lr_scheduler import LambdaLR


def learning_rate_fn(current_step: int, warmup_steps: int, final_step: int, final_ratio: float) -> float:
    """Factor the peak learning rate is multiplied with at current_step.

    Linear warmup from 0 to 1 over warmup_steps, then decay to final_ratio at final_step (constant afterwards).
    NB: the decay is not a plain cosine: the cosine is multiplied by a factor that itself decays linearly,
    so the learning rate drops faster than a standard cosine schedule. Kept as is for reproducibility.
    """
    if current_step < warmup_steps:
        return float(current_step) / float(warmup_steps)
    elif current_step < final_step:
        steps_remaining = final_step - current_step
        steps_after_warmup = current_step - warmup_steps
        total_steps_after_warmup = final_step - warmup_steps

        factor = (1 - final_ratio) * (steps_remaining + steps_after_warmup * final_ratio) / total_steps_after_warmup
        return final_ratio + factor * (1 + math.cos(math.pi * steps_after_warmup / total_steps_after_warmup)) / 2

    return final_ratio


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    _name_: str,
    warmup_steps: int,
    final_step: int,
    final_ratio: float,
    **kwargs,
) -> LambdaLR:
    """Learning-rate scheduler; only "CosineDecay" (see learning_rate_fn) is supported."""
    if _name_ != "CosineDecay":
        raise ValueError(f"Unsupported scheduler: {_name_}")
    return LambdaLR(
        optimizer,
        partial(learning_rate_fn, warmup_steps=warmup_steps, final_step=final_step, final_ratio=final_ratio),
    )
