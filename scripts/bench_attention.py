import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model import ModelConfig
from llm125m.utils.logging import setup_logger


def naive_causal_attention(q, k, v):
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    scores.masked_fill_(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def sdpa_causal_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def benchmark_kernel(fn, q, k, v, warmup, iters):
    for _ in range(warmup):
        out = fn(q, k, v)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        out = fn(q, k, v)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.time() - t0) / iters
    return elapsed, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--log-file", type=str, default="logs/bench_attention.log")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    logger = setup_logger("bench_attention", log_file=args.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    mcfg = ModelConfig()
    n_q = mcfg.n_q_heads
    head_dim = mcfg.head_dim

    logger.info(f"device={device} dtype={dtype} n_q_heads={n_q} head_dim={head_dim}")

    for seq_len in args.seq_lens:
        for batch_size in args.batch_sizes:
            try:
                q = torch.randn(batch_size, n_q, seq_len, head_dim, device=device, dtype=dtype)
                k = torch.randn(batch_size, n_q, seq_len, head_dim, device=device, dtype=dtype)
                v = torch.randn(batch_size, n_q, seq_len, head_dim, device=device, dtype=dtype)

                t_sdpa, out_sdpa = benchmark_kernel(sdpa_causal_attention, q, k, v,
                                                    args.warmup, args.iters)
                t_naive, out_naive = benchmark_kernel(naive_causal_attention, q, k, v,
                                                      args.warmup, args.iters)

                max_diff = (out_sdpa.float() - out_naive.float()).abs().max().item()
                speedup = t_naive / max(1e-9, t_sdpa)

                logger.info(
                    f"bench batch={batch_size} seq_len={seq_len} "
                    f"sdpa_ms={t_sdpa*1000:.3f} naive_ms={t_naive*1000:.3f} "
                    f"speedup={speedup:.2f}x max_abs_diff={max_diff:.4e}"
                )

                del q, k, v, out_sdpa, out_naive
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"oom batch={batch_size} seq_len={seq_len}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
