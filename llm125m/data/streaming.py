import glob
import numpy as np
import torch
from torch.utils.data import IterableDataset


class TokenShardDataset(IterableDataset):
    def __init__(self, shard_pattern: str, seq_len: int, shuffle_shards: bool = True, seed: int = 42):
        self.shard_files = sorted(glob.glob(shard_pattern))
        if not self.shard_files:
            raise FileNotFoundError(f"no shards matched pattern {shard_pattern}")
        self.seq_len = seq_len
        self.shuffle_shards = shuffle_shards
        self.seed = seed

    def num_shards(self) -> int:
        return len(self.shard_files)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = list(self.shard_files)
            worker_id = 0
        else:
            files = self.shard_files[worker_info.id::worker_info.num_workers]
            worker_id = worker_info.id

        rng = np.random.default_rng(self.seed + worker_id)
        if self.shuffle_shards:
            rng.shuffle(files)

        for f in files:
            data = np.memmap(f, dtype=np.uint16, mode="r")
            n_samples = (len(data) - 1) // self.seq_len
            indices = np.arange(n_samples)
            if self.shuffle_shards:
                rng.shuffle(indices)
            for idx in indices:
                start = int(idx) * self.seq_len
                chunk = data[start:start + self.seq_len + 1].astype(np.int64)
                tokens = torch.from_numpy(chunk[:-1])
                targets = torch.from_numpy(chunk[1:])
                yield tokens, targets
