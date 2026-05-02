import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model import ModelConfig
from llm125m.model import Transformer
from llm125m.utils.logging import setup_logger


def benchmark_generation(model, batch_size, prompt_len, new_tokens, vocab_size, device,
                         temperature, top_k, top_p, warmup, iters):
    model.eval()
    prompt = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)

    for _ in range(warmup):
        _ = model.generate(prompt, max_new_tokens=8, temperature=temperature,
                           top_k=top_k, top_p=top_p)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    total_new = 0
    for _ in range(iters):
        out = model.generate(prompt, max_new_tokens=new_tokens, temperature=temperature,
                             top_k=top_k, top_p=top_p)
        total_new += (out.size(1) - prompt_len) * batch_size
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return total_new / max(1e-9, elapsed), elapsed / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--log-file", type=str, default="logs/bench_generation.log")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    logger = setup_logger("bench_generation", log_file=args.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = ModelConfig()

    logger.info(f"device={device} prompt_len={args.prompt_len} new_tokens={args.new_tokens}")

    model = Transformer(mcfg).to(device)
    if torch.cuda.is_available():
        model = model.half()

    sampling_configs = [
        ("greedy", 1.0, 0, 0.0),
        ("topk_50_t08", 0.8, 50, 0.0),
        ("topp_09_t08", 0.8, 0, 0.9),
    ]

    for batch_size in args.batch_sizes:
        for label, temp, tk, tp in sampling_configs:
            try:
                tk_arg = tk if tk > 0 else None
                tp_arg = tp if tp > 0 else None
                tokens_per_sec, latency_s = benchmark_generation(
                    model, batch_size, args.prompt_len, args.new_tokens,
                    mcfg.vocab_size, device, temp, tk_arg, tp_arg,
                    args.warmup, args.iters,
                )
                logger.info(
                    f"bench batch={batch_size} sampling={label} "
                    f"latency_s={latency_s:.3f} tokens_per_sec={tokens_per_sec:.0f} "
                    f"ms_per_token={1000.0/max(1e-9, tokens_per_sec):.2f}"
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"oom batch={batch_size} sampling={label}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
