import torch
from torch import Tensor
from typing import List, Tuple

from ..tokenizer import ProteinTokenizer

from discrete_flow_matching.noise_scheduler import noising


class DataCollatorDFMLM(object):
    """
    Turns a list of (record_id, sequence, class_label) CSV rows into a noised training batch.

    Args:
        tokenizer: ProteinTokenizer used to encode the sequences.
        max_length: maximum sequence length; longer sequences are truncated.
        random_truncate: truncate to a random subsequence rather than the first max_length tokens.
        return_labels: whether to also return the record ids (not compatible with accelerate).
        diffusion_type: 'uniform' or 'masking'.
        scheduler_type, polynomial_power: noise schedule kappa(t); the power is only used by
            (inv_)polynomial (see noise_scheduler).
        t_min_train, t_max_train: training times are drawn from [t_min_train, t_max_train].
        padding: pad the batch to the "longest" sequence in it, or to "max_length".
        pad_to_multiple_of: pad the length to a multiple of this (for memory-efficient attention).
        dtype: dtype of the additive pad_mask; must match the autocast dtype of the forward pass.

    Returns (x, t, y, pad_mask, class_labels), or with the record ids prepended if return_labels.
    x and y are [B, L]; y is the clean sequence, set to -100 on <pad>/<bos>/<eos> (and, under
    masking, on the uncorrupted positions) so the loss ignores them. pad_mask is additive:
    0 on real tokens, -inf on padding.
    """

    def __init__(
        self,
        tokenizer: ProteinTokenizer,
        max_length: int,
        random_truncate: bool,
        return_labels: bool,
        diffusion_type: str,
        scheduler_type: str,
        polynomial_power: int,
        t_max_train: float,
        t_min_train: float,
        padding: str,
        pad_to_multiple_of: int,
        dtype: torch.dtype,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.random_truncate = random_truncate
        self.return_labels = return_labels
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.dtype = dtype

        self.diffusion_type = diffusion_type
        self.scheduler_type = scheduler_type
        self.polynomial_power = polynomial_power
        self.t_max_train = t_max_train
        self.t_min_train = t_min_train

    def __call__(self, inputs: List[Tuple]) -> Tuple[Tensor, ...]:
        """
        Inspired by
        https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/data/data_collator.py#L607
        """
        labels, proteins, class_labels = zip(*inputs)
        class_labels = torch.tensor([int(c) for c in class_labels], dtype=torch.long)

        # Tokenize the inputs
        proteins = [self.tokenizer.encode(p, self.max_length, random_truncate=self.random_truncate) for p in proteins]

        # Compute the length of the batch min(longest sequence, max_length)
        if self.padding == "longest":
            max_length = max(p.size(0) for p in proteins)
        elif self.padding == "max_length":
            max_length = self.max_length

        # Pad the sequence to a multiple of the provided value. Necessary for memory efficient attention
        if self.pad_to_multiple_of is not None and (max_length % self.pad_to_multiple_of != 0):
            max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        # Initialize the input tensor with padding, then stack the encoded sequences into it
        x = torch.full(size=(len(proteins), max_length), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        for i, p in enumerate(proteins):
            x[i, : p.shape[0]] = p

        # Compute the padding, <bos> and <eos> masks
        pad_mask = x == self.tokenizer.pad_token_id
        bos_mask = x == self.tokenizer.bos_token_id
        eos_mask = x == self.tokenizer.eos_token_id
        special_tokens_mask = pad_mask | bos_mask | eos_mask

        # Create the label tensor
        y = x.clone()

        # Antithetic time sampling: one random offset, evenly spaced over the batch (as in MD4).
        t = torch.rand(1, device=x.device) + (1 + torch.arange(x.shape[0], device=x.device)).float() / x.shape[0]
        t = torch.fmod(t, 1)  # Wrap around using modulo 1.
        t = self.t_min_train + t * (self.t_max_train - self.t_min_train)
        x, corrupted_mask = noising(x, t, tokenizer=self.tokenizer, diffusion_type=self.diffusion_type,
                                    scheduler_type=self.scheduler_type, polynomial_power=self.polynomial_power)

        # The loss is only computed on the amino-acid positions (-100 is PyTorch's ignore_index)
        y[special_tokens_mask] = -100
        # Under masking it is further restricted to the corrupted tokens
        if self.diffusion_type.lower() == "masking":
            y[~corrupted_mask] = -100

        # The model takes an additive pad_mask, added straight to the attention scores
        pad_mask = torch.where(pad_mask, float("-inf"), float(0.0)).type(self.dtype)

        if self.return_labels:
            return labels, x, t, y, pad_mask, class_labels
        return x, t, y, pad_mask, class_labels
