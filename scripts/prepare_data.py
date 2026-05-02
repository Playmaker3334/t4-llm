import argparse
import os
import sys
import time

import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.data import DataConfig
from llm125m.data.tokenizer import load_tokenizer
from llm125m.data.packing import ShardWriter
from llm125m.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=10_000_000_000)
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--eval-tokens", type=int, default=50_000_000)
    parser.add_argument("--log-file", type=str, default="logs/prepare_data.log")
    args = parser.parse_args()

    cfg = DataConfig()
    output_dir = args.output_dir or cfg.shard_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)

    logger = setup_logger("prepare_data", log_file=args.log_file)
    logger.info(f"config dataset={cfg.dataset_path} name={cfg.dataset_name} "
                f"tokenizer={cfg.tokenizer_path} output_dir={output_dir} "
                f"max_tokens={args.max_tokens} shard_size={args.shard_size} "
                f"eval_tokens={args.eval_tokens}")

    tokenizer = load_tokenizer(cfg.tokenizer_path)
    eos_id = tokenizer.eos_token_id
    logger.info(f"tokenizer_loaded vocab_size={tokenizer.vocab_size} eos_id={eos_id}")

    ds = load_dataset(cfg.dataset_path, name=cfg.dataset_name, split="train", streaming=True)

    eval_buf = np.zeros(args.eval_tokens, dtype=np.uint16)
    eval_idx = 0
    eval_done = False

    train_writer = ShardWriter(output_dir, "train", args.shard_size)

    t_start = time.time()
    last_log = t_start
    docs_processed = 0

    for example in ds:
        text = example["text"]
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos_id)
        docs_processed += 1

        cursor = 0
        if not eval_done:
            remaining = args.eval_tokens - eval_idx
            take = min(len(ids), remaining)
            eval_buf[eval_idx:eval_idx + take] = ids[:take]
            eval_idx += take
            cursor = take
            if eval_idx >= args.eval_tokens:
                eval_done = True
                logger.info(f"eval_buffer_complete tokens={eval_idx}")

        if eval_done and cursor < len(ids):
            train_writer.add(ids[cursor:])
            result = train_writer.flush() if False else None

        now = time.time()
        if now - last_log > 30.0:
            elapsed = now - t_start
            tps = train_writer.total / max(1.0, elapsed)
            logger.info(f"progress docs={docs_processed} train_tokens={train_writer.total} "
                        f"shards_written={train_writer.shard_idx} tokens_per_sec={tps:.0f}")
            last_log = now

        if train_writer.total >= args.max_tokens:
            break

    last = train_writer.close()
    if last is not None:
        path, n = last
        logger.info(f"shard_written path={path} tokens={n}")

    eval_path = os.path.join(output_dir, "eval_00000.bin")
    eval_buf[:eval_idx].tofile(eval_path)
    logger.info(f"eval_written path={eval_path} tokens={eval_idx}")

    elapsed = time.time() - t_start
    logger.info(f"complete total_train_tokens={train_writer.total} "
                f"shards={train_writer.shard_idx} eval_tokens={eval_idx} "
                f"docs={docs_processed} elapsed_s={elapsed:.1f}")


if __name__ == "__main__":
    main()
