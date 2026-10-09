from typing import Tuple, Iterator
from itertools import islice, zip_longest, repeat, chain
from torch.utils.data import IterableDataset, get_worker_info


class IterableProteinDataset(IterableDataset):
    """Streams (record_id, sequence, class_label) rows from one or more CSV files.

    The files are read lazily rather than loaded into memory: the training split is tens of
    millions of sequences. With several dataloader workers, each worker takes every
    num_workers-th row (offset by its own id), so the rows are sharded without overlap.
    """

    def __init__(self, paths: list, samples_before_next_set: list | None):
        """
        Args:
            paths (list): Paths to the CSV files to read.
            samples_before_next_set (list | None): Number of samples of each dataset to return before moving to the
            next dataset (interleaving).
        """
        self.paths = paths
        self.samples_per_set = samples_before_next_set if samples_before_next_set is not None else [1] * len(paths)

    def parse_file(self) -> Iterator[Tuple[str, str, str]]:
        worker_info = get_worker_info()
        step = 1 if worker_info is None else worker_info.num_workers
        offset = 0 if worker_info is None else worker_info.id

        files, iterator = [], []
        for path, n in zip(self.paths, self.samples_per_set):
            # Open the file
            file = open(path, "r")
            # Skip header
            next(file)
            # Add the file to the list of files to close them at the end
            files.append(file)
            # Add the file iterator to the list of iterators n times
            iterator.extend(repeat(file, n))

        # Interleave the iterators and pad with None
        iterator = chain.from_iterable(zip_longest(*iterator, fillvalue=None))

        # Iterate through the datasets
        for row in islice(iterator, offset, None, step):
            if row is not None:
                # Rows are (record_id, sequence, class_label)
                yield row.strip().split(",")

        # Closing the files
        for file in files:
            file.close()

    def __iter__(self) -> Iterator[Tuple[str, str, str]]:
        return self.parse_file()
