import os
import random

import numpy as np
import torch
import torch.nn.functional as F


def to_one_hot(x, vocab_size):
    """One-hot encoding of the sequence tokens: [B, L] -> [B, L, vocab_size] (float)."""
    return F.one_hot(x.long(), num_classes=vocab_size).float()


def get_special_token_mask(x, tokenizer):
    """
    Splits the positions of x [B, L] by their role in the diffusion.

    Returns:
        is_not_special (Tensor): [B, L] True on amino-acid positions, False on <pad>/<bos>/<eos>/
            <mask> -- i.e. the positions that may be resampled under uniform noise.
        is_masked (Tensor): [B, L] True on <mask> -- the positions still to be revealed under
            masking noise.
        special_ids (Tensor): the special token ids, needed to zero them out of a sampling
            distribution (see sampling_utils.denoising_step).
    """
    special_ids_tensor = torch.tensor(list(tokenizer.special_token_ids), device=x.device)
    is_not_special = ~torch.isin(x, special_ids_tensor)
    is_masked = (x == tokenizer.mask_token_id)
    return is_not_special, is_masked, special_ids_tensor


def set_all_seeds(seed=42):
    """Seed Python, NumPy and PyTorch (CPU and all GPUs) and make cuDNN deterministic."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN kernels (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
