import argparse
import os
import sys
import time

import torch
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model import ModelConfig
from configs.training import TrainConfig
from llm125m.model import Transformer
from llm125m.training.muon import Muon, newton_schulz_orthogonalize
from llm125m.training.optimizer import build_optimizers
from llm125m.utils.logging import setup_logger


def time_optimizer_step(opt, params, warmup, iters):
    for p in params:
        if p.grad is None:
            p.grad = torch.randn_like(p)
        else:
            p.grad.normal_()

    for _ in range(warmup):
        opt.step()
        for p in params:
            p.grad.normal_()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        opt.step()
        for p in params:
            p.grad.normal_()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / iters


def benchmark_newton_schulz(device, warmup=3, iters=20):
    sizes = [(576, 576), (576, 1536), (1536, 576), (5184, 576), (576, 5184)]
    results = []
    for d0, d1 in sizes:
        x = torch.randn(d0, d1, device=device)
        for _ in range(warmup):
            _ = newton_schulz_orthogonalize(x, steps=5)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = newton_schulz_orthogonalize(x, steps=5)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        avg_ms = (time.time() - t0) / iters * 1000
        results.append((d0, d1, avg_ms))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--log-file", type=str, default="logs/bench_optimizer.log")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    logger = setup_logger("bench_optimizer", log_file=args.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    mcfg = ModelConfig()
    tcfg = TrainConfig()

    model = Transformer(mcfg).to(device)
    muon_opt, adamw_opt, matrix_params, other_params = build_optimizers(model, tcfg)

    n_muon = sum(p.numel() for p in matrix_params)
    n_adamw = sum(p.numel() for p in other_params)
    logger.info(f"split muon_tensors={len(matrix_params)} muon_params={n_muon:,} "
                f"adamw_tensors={len(other_params)} adamw_params={n_adamw:,}")

    muon_step_s = time_optimizer_step(muon_opt, matrix_params, args.warmup, args.iters)
    adamw_step_s = time_optimizer_step(adamw_opt, other_params, args.warmup, args.iters)

    logger.info(f"step_time muon_ms={muon_step_s*1000:.3f} adamw_ms={adamw_step_s*1000:.3f} "
                f"ratio={muon_step_s/max(1e-9, adamw_step_s):.2f}x")

    only_adamw = AdamW(matrix_params, lr=tcfg.adamw_lr, betas=tcfg.adamw_betas,
                       eps=tcfg.adamw_eps, weight_decay=tcfg.weight_decay_matrices)
    adamw_on_matrices_s = time_optimizer_step(only_adamw, matrix_params, args.warmup, args.iters)
    logger.info(f"comparison adamw_on_matrices_ms={adamw_on_matrices_s*1000:.3f} "
                f"muon_overhead_vs_adamw={(muon_step_s - adamw_on_matrices_s)*1000:.3f}ms")

    logger.info("newton_schulz_kernel_only")
    ns_results = benchmark_newton_schulz(device, warmup=3, iters=20)
    for d0, d1, avg_ms in ns_results:
        logger.info(f"  shape=({d0},{d1}) avg_ms={avg_ms:.3f}")


if __name__ == "__main__":
    main()
