import torch
import torch.nn.functional as F

import numpy as np
import random
import os

def to_one_hot(x, vocab_size):
    """One-hot encoding of the pixel tokens: [B, C, H, W] -> [B, C, H, W, vocab_size] (float)."""
    return F.one_hot(x.long(), num_classes=vocab_size).float()

def get_masked_pixels(x, vocab_size):
    """Boolean mask of the pixels equal to the MASK token (the last vocabulary index)."""
    is_masked = (x == vocab_size-1)
    return is_masked

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
