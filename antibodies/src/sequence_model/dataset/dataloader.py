import torch
from torch.utils.data import DataLoader

from ..tokenizer import ProteinTokenizer

from .iterable_protein_dataset import IterableProteinDataset
from .data_collator import DataCollatorDFMLM


def get_dataloader(
    vocab_path: str,
    pad_token_id: int,
    mask_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    unk_token_id: int,
    other_special_token_ids: list | None,
    paths: dict,
    max_length: int,
    random_truncate: bool,
    return_labels: bool,
    diffusion_type: str,
    scheduler_type: str,
    polynomial_power: int,
    t_max_train: float,
    t_min_train: float,
    num_workers: int,
    per_device_batch_size: int,
    samples_before_next_set: list | None = None,
    padding: str = "max_length",
    pad_to_multiple_of: int = 8,
    dtype: torch.dtype = torch.float32,
    merge: bool = False,
    **kwargs,
) -> DataLoader:
    """Builds the training / validation dataloaders over the (record_id, sequence, label) CSVs.

    Args:
        vocab_path (str): Path to the vocabulary file to load.
        pad_token_id, mask_token_id, bos_token_id, eos_token_id, unk_token_id (int): Token
            indices in the vocab file.
        other_special_token_ids (list | None): Indices of any further special tokens.
        paths (dict): Dict of name:path to the CSV files to read.
        max_length (int): Maximum sequence length.
        random_truncate (bool): Truncate to a random subsequence if longer than max_length.
        return_labels (bool): Also return the record ids.
        diffusion_type, scheduler_type, polynomial_power, t_min_train, t_max_train: noise
            process settings, passed through to the collator.
        num_workers (int): Number of dataloader workers.
        per_device_batch_size (int): Batch size for each GPU.
        samples_before_next_set (list | None): Number of samples to take from each dataset
            before moving to the next (interleaving). Defaults to ``None``.
        padding (str): Pad the batch to the "longest" sequence or to "max_length".
        pad_to_multiple_of (int): Pad the length to a multiple of this. Defaults to 8.
        dtype (torch.dtype): Dtype of the additive pad_mask. Defaults to torch.float32.
        merge (bool): Interleave every path into one DataLoader (training), rather than
            returning one DataLoader per path (validation, which is reported per set).

    Returns:
        torch.utils.data.DataLoader if merge, else a dict of name:DataLoader.
    """
    tokenizer = ProteinTokenizer(
        vocab_path=vocab_path,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        unk_token_id=unk_token_id,
        other_special_token_ids=other_special_token_ids,
    )

    collator = DataCollatorDFMLM(
        tokenizer=tokenizer,
        max_length=max_length,
        random_truncate=random_truncate,
        return_labels=return_labels,
        diffusion_type=diffusion_type,
        scheduler_type=scheduler_type,
        polynomial_power=polynomial_power,
        t_max_train=t_max_train,
        t_min_train=t_min_train,
        padding=padding,
        pad_to_multiple_of=pad_to_multiple_of,
        dtype=dtype,
    )

    loader_kwargs = dict(
        collate_fn=collator,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )

    if merge:
        return DataLoader(
            IterableProteinDataset(paths.values(), samples_before_next_set),
            per_device_batch_size,
            **loader_kwargs,
        )
    return {
        k: DataLoader(
            IterableProteinDataset([v], samples_before_next_set),
            per_device_batch_size,
            **loader_kwargs,
        )
        for k, v in paths.items()
    }
