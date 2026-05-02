import os
import numpy as np


class ShardWriter:
    def __init__(self, output_dir: str, prefix: str, shard_size: int):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.buffer = np.zeros(shard_size, dtype=np.uint16)
        self.idx = 0
        self.shard_idx = 0
        self.total = 0
        os.makedirs(output_dir, exist_ok=True)

    def add(self, tokens):
        for tok in tokens:
            if self.idx >= self.shard_size:
                self.flush()
            self.buffer[self.idx] = tok
            self.idx += 1
            self.total += 1

    def flush(self):
        if self.idx == 0:
            return None
        path = os.path.join(self.output_dir, f"{self.prefix}_{self.shard_idx:05d}.bin")
        self.buffer[:self.idx].tofile(path)
        written = self.idx
        self.shard_idx += 1
        self.idx = 0
        return path, written

    def close(self):
        return self.flush()
