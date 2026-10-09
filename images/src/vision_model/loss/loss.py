from torch.nn import CrossEntropyLoss

from discrete_flow_matching.noise_scheduler import get_scheduler_images as get_scheduler


def get_loss(label_smoothing: float = 0.0, **kwargs) -> CrossEntropyLoss:
    """Per-token cross-entropy (reduction='none'), ignoring targets equal to -100."""
    return CrossEntropyLoss(reduction="none", ignore_index=-100, label_smoothing=label_smoothing)


def get_loss_weight(t, max_weight_loss, scheduler_type, polynomial_power=None):
    """
    Time weighting of the loss, kappa_dot(t) / (1 - kappa(t)), as in the MD4 ELBO
    (Shi et al., 2024, https://arxiv.org/abs/2406.04329). The 1 / max_weight_loss term in the
    denominator keeps the weight finite as kappa(t) -> 1.
    """
    kappa, kappa_dot = get_scheduler(t, scheduler_type=scheduler_type, polynomial_power=polynomial_power)
    epsilon = 1 / max_weight_loss
    return kappa_dot / (1 - kappa + epsilon)
