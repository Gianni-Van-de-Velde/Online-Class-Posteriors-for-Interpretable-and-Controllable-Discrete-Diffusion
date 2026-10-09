__all__ = [
    "sample",
    "sample_with_posterior",
    "sample_NOT",
]

from .sampling_standard import sample, sample_with_posterior
from .sampling_logical_operators import sample_NOT
