import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model import ModelConfig
from configs.training import TrainConfig
from llm125m.model import Transformer
from llm125m.training.optimizer import build_optimizers
from llm125m.utils.logging import setup_logger


def measure_memory(batch_size, seq_len, vocab_size, mcfg, tcfg, device, fp16, mode):
    model = Transformer(mcfg).to(device)
    muon_opt, adamw_opt, _, _ = build_optimizers(model, tcfg)
    scaler = GradScaler(enabled=fp16, init_scale=tcfg.grad_scaler_init)
    model.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    if mode in ("forward", "full"):
        with autocast(dtype=torch.float16, enabled=fp16):
            _, loss = model(tokens, targets)

    if mode == "full":
        for opt in [muon_opt, adamw_opt]:
            opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        for opt in [muon_opt, adamw_opt]:
            scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in [muon_opt, adamw_opt]:
            scaler.step(opt)
        scaler.update()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        alloc_mb = torch.cuda.memory_allocated() / 1024 / 1024
    else:
        peak_mb = 0.0
        alloc_mb = 0.0

    del model, muon_opt, adamw_opt, scaler, tokens, targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return peak_mb, alloc_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--mode", type=str, choices=["forward", "full"], default="full")
    parser.add_argument("--log-file", type=str, default="logs/bench_memory.log")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    logger = setup_logger("bench_memory", log_file=args.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logger.warning("cuda_unavailable memory_benchmark_uninformative")

    logger.info(f"device={device} mode={args.mode}")

    mcfg = ModelConfig()
    tcfg = TrainConfig()

    for seq_len in args.seq_lens:
        if seq_len > mcfg.max_seq_len:
            logger.warning(f"skipping seq_len={seq_len} exceeds max_seq_len={mcfg.max_seq_len}")
            continue

        for batch_size in args.batch_sizes:
            tcfg.seq_len = seq_len
            tcfg.micro_batch_size = batch_size
            try:
                peak, alloc = measure_memory(
                    batch_size, seq_len, mcfg.vocab_size,
                    mcfg, tcfg, device, tcfg.fp16, args.mode,
                )
                logger.info(
                    f"memory batch={batch_size} seq_len={seq_len} "
                    f"peak_mb={peak:.0f} resident_mb={alloc:.0f}"
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"oom batch={batch_size} seq_len={seq_len}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
