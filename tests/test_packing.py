import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm125m.data.streaming import TokenShardDataset
from llm125m.data.packing import ShardWriter


def make_shards(tmp_dir, n_shards=3, shard_size=1024):
    rng = np.random.default_rng(123)
    paths = []
    for i in range(n_shards):
        path = os.path.join(tmp_dir, f"train_{i:05d}.bin")
        data = rng.integers(0, 32000, shard_size, dtype=np.uint16)
        data.tofile(path)
        paths.append(path)
    return paths


def test_streaming_yields_correct_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        make_shards(tmp)
        ds = TokenShardDataset(os.path.join(tmp, "train_*.bin"), seq_len=128, shuffle_shards=False)
        items = list(ds)
        assert len(items) > 0
        for tokens, targets in items:
            assert tokens.shape == (128,)
            assert targets.shape == (128,)
            assert torch.equal(tokens[1:], targets[:-1])


def test_streaming_reproducible_with_same_seed():
    with tempfile.TemporaryDirectory() as tmp:
        make_shards(tmp)
        ds1 = TokenShardDataset(os.path.join(tmp, "train_*.bin"), seq_len=64, shuffle_shards=True, seed=7)
        ds2 = TokenShardDataset(os.path.join(tmp, "train_*.bin"), seq_len=64, shuffle_shards=True, seed=7)
        items1 = [t.clone() for t, _ in ds1]
        items2 = [t.clone() for t, _ in ds2]
        assert len(items1) == len(items2)
        for a, b in zip(items1, items2):
            assert torch.equal(a, b)


def test_streaming_different_seed_changes_order():
    with tempfile.TemporaryDirectory() as tmp:
        make_shards(tmp)
        ds1 = TokenShardDataset(os.path.join(tmp, "train_*.bin"), seq_len=64, shuffle_shards=True, seed=1)
        ds2 = TokenShardDataset(os.path.join(tmp, "train_*.bin"), seq_len=64, shuffle_shards=True, seed=99)
        items1 = [t.clone() for t, _ in ds1]
        items2 = [t.clone() for t, _ in ds2]
        any_different = any(not torch.equal(a, b) for a, b in zip(items1, items2))
        assert any_different


def test_streaming_raises_when_no_shards():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            TokenShardDataset(os.path.join(tmp, "missing_*.bin"), seq_len=64)
        except FileNotFoundError:
            return
    assert False, "expected FileNotFoundError"


def test_shard_writer_creates_correct_files():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(tmp, "train", shard_size=512)
        writer.add(list(range(2000)))
        writer.close()
        files = sorted(os.listdir(tmp))
        assert len(files) >= 3
        assert all(f.startswith("train_") and f.endswith(".bin") for f in files)
