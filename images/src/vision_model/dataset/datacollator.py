import torch
import numpy as np

from discrete_flow_matching.noise_scheduler import noising_images


class DatacollatorImages:
    """
    Turns a list of {"img" | "image", "label"} examples into a noised training batch.

    Args:
        return_labels: whether to also return the class labels.
        diffusion_type: 'uniform' or 'masking'.
        scheduler_type, polynomial_power: noise schedule kappa(t); the power is only used by (inv_)polynomial (see noise_scheduler).
        t_min_train, t_max_train: training times are drawn from [t_min_train, t_max_train].
        vocab_size: number of tokens, including the mask token under masking (mask = vocab_size - 1).

    Returns (labels, x_t, t, y), or (x_t, t, y) if not return_labels, with x_t and y of shape
    [B, C, H, W]. y is the clean image; under masking it is -100 (ignored) on unmasked pixels.
    """

    def __init__(
        self,
        return_labels: bool,
        diffusion_type: str,
        scheduler_type: str,
        polynomial_power: float | None,
        t_max_train: float,
        t_min_train: float,
        vocab_size: int,
    ) -> None:
        self.return_labels = return_labels
        self.diffusion_type = diffusion_type
        self.scheduler_type = scheduler_type
        self.polynomial_power = polynomial_power
        self.t_max_train = t_max_train
        self.t_min_train = t_min_train
        self.vocab_size = vocab_size

    def __call__(self, examples):
        def _to_hwc(ex):
            img = np.array(ex.get("img", ex.get("image")))
            if img.ndim == 2:  # grayscale
                img = img[:, :, None]
            return torch.tensor(img).long()

        x_0 = torch.stack([_to_hwc(ex) for ex in examples])  # [B, H, W, C]
        x_0 = x_0.permute(0, 3, 1, 2).contiguous()           # [B, C, H, W]
        labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
        y = x_0.clone()

        # Antithetic time sampling: one random offset, evenly spaced over the batch.
        bs = x_0.shape[0]
        device = x_0.device
        t = torch.rand(1, device=device) + (1 + torch.arange(bs, device=device)).float() / bs
        t = torch.fmod(t, 1.0)
        t = self.t_min_train + t * (self.t_max_train - self.t_min_train)

        x_t, corrupted_mask = noising_images(
            x_0,
            t,
            self.vocab_size,
            diffusion_type=self.diffusion_type,
            scheduler_type=self.scheduler_type,
            polynomial_power=self.polynomial_power,
        )

        # Under masking the loss is only computed on the masked pixels.
        if self.diffusion_type.lower() == "masking":
            y[~corrupted_mask] = -100

        return (labels, x_t, t, y) if self.return_labels else (x_t, t, y)
