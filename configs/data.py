from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_path: str = "HuggingFaceFW/fineweb-edu"
    dataset_name: str = "sample-10BT"
    tokenizer_path: str = "meta-llama/Llama-2-7b-hf"
    shard_dir: str = "data/shards"
    train_pattern: str = "train_*.bin"
    eval_pattern: str = "eval_*.bin"
    num_workers: int = 2
    prefetch_factor: int = 2
    shard_size_tokens: int = 100_000_000
    eval_tokens: int = 50_000_000
