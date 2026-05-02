import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model import ModelConfig
from configs.training import TrainConfig
from llm125m.model import Transformer
from llm125m.training.optimizer import build_optimizers
from llm125m.utils.logging import setup_logger
from llm125m.utils.throughput import T4_PEAK_FLOPS_FP16


def benchmark_config(model, optimizers, scaler, batch_size, seq_len, vocab_size,
                     device, fp16, warmup, iters, logger):
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    for _ in range(warmup):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16, enabled=fp16):
            _, loss = model(tokens, targets)
        scaler.scale(loss).backward()
        for opt in optimizers:
            scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_fwd = 0.0
    t_bwd = 0.0
    t_step = 0.0
    for _ in range(iters):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with autocast(dtype=torch.float16, enabled=fp16):
            _, loss = model(tokens, targets)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        t_fwd += (t1 - t0)

        scaler.scale(loss).backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()
        t_bwd += (t2 - t1)

        for opt in optimizers:
            scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.time()
        t_step += (t3 - t2)

    avg_fwd = t_fwd / iters
    avg_bwd = t_bwd / iters
    avg_step = t_step / iters
    avg_total = avg_fwd + avg_bwd + avg_step

    tokens_per_step = batch_size * seq_len
    tokens_per_sec = tokens_per_step / avg_total

    return {
        "fwd_ms": avg_fwd * 1000,
        "bwd_ms": avg_bwd * 1000,
        "step_ms": avg_step * 1000,
        "total_ms": avg_total * 1000,
        "tokens_per_sec": tokens_per_sec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--log-file", type=str, default="logs/bench_throughput.log")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    logger = setup_logger("bench_throughput", log_file=args.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    mcfg = ModelConfig()
    tcfg = TrainConfig()

    flops_per_token_const = 6.0 * 125_000_000

    for seq_len in args.seq_lens:
        if seq_len > mcfg.max_seq_len:
            logger.warning(f"skipping seq_len={seq_len} exceeds max_seq_len={mcfg.max_seq_len}")
            continue

        for batch_size in args.batch_sizes:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            model = Transformer(mcfg).to(device)
            tcfg.seq_len = seq_len
            tcfg.micro_batch_size = batch_size
            muon_opt, adamw_opt, _, _ = build_optimizers(model, tcfg)
            scaler = GradScaler(enabled=tcfg.fp16, init_scale=tcfg.grad_scaler_init)
            model.train()

            try:
                result = benchmark_config(
                    model, [muon_opt, adamw_opt], scaler,
                    batch_size, seq_len, mcfg.vocab_size,
                    device, tcfg.fp16, args.warmup, args.iters, logger,
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"oom batch={batch_size} seq_len={seq_len}")
                del model, muon_opt, adamw_opt, scaler
                continue

            flops_per_token = flops_per_token_const + 12.0 * mcfg.n_layers * mcfg.hidden_dim * seq_len
            achieved_flops = flops_per_token * result["tokens_per_sec"]
            mfu = achieved_flops / T4_PEAK_FLOPS_FP16

            mem_mb = 0.0
            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()

            logger.info(
                f"bench batch={batch_size} seq_len={seq_len} "
                f"fwd_ms={result['fwd_ms']:.2f} bwd_ms={result['bwd_ms']:.2f} "
                f"step_ms={result['step_ms']:.2f} total_ms={result['total_ms']:.2f} "
                f"tokens_per_sec={result['tokens_per_sec']:.0f} "
                f"mfu={mfu:.3f} peak_mem_mb={mem_mb:.0f}"
            )

            del model, muon_opt, adamw_opt, scaler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
