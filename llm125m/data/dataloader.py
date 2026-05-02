import os
from torch.utils.data import DataLoader

from llm125m.data.streaming import TokenShardDataset


def build_train_loader(data_config, train_config):
    pattern = os.path.join(data_config.shard_dir, data_config.train_pattern)
    dataset = TokenShardDataset(
        shard_pattern=pattern,
        seq_len=train_config.seq_len,
        shuffle_shards=True,
        seed=train_config.seed,
    )
    return DataLoader(
        dataset,
        batch_size=train_config.micro_batch_size,
        num_workers=data_config.num_workers,
        pin_memory=True,
        prefetch_factor=data_config.prefetch_factor if data_config.num_workers > 0 else None,
        drop_last=True,
    )


def build_eval_loader(data_config, train_config):
    pattern = os.path.join(data_config.shard_dir, data_config.eval_pattern)
    dataset = TokenShardDataset(
        shard_pattern=pattern,
        seq_len=train_config.seq_len,
        shuffle_shards=False,
    )
    return DataLoader(
        dataset,
        batch_size=train_config.micro_batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
