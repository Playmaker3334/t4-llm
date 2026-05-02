import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model import ModelConfig
from configs.training import TrainConfig
from llm125m.model import Transformer
from llm125m.training.optimizer import build_optimizers
from llm125m.utils.logging import setup_logger
from llm125m.utils.diagnostics import TrainingDiagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    logger = setup_logger("smoke_test")

    mcfg = ModelConfig()
    tcfg = TrainConfig()
    tcfg.seq_len = args.seq_len
    tcfg.micro_batch_size = args.batch_size

    device = torch.device(args.device)
    logger.info(f"device={device} batch={args.batch_size} seq_len={args.seq_len}")

    model = Transformer(mcfg).to(device)
    n_total = model.num_parameters()
    n_excl = model.num_parameters(exclude_embeddings=True)
    expected_low, expected_high = 120_000_000, 130_000_000

    if not (expected_low < n_total < expected_high):
        logger.error(f"param_count_unexpected total={n_total:,} expected~125M")
    else:
        logger.info(f"param_count_ok total={n_total:,} excl_emb={n_excl:,}")

    muon_opt, adamw_opt, matrix_params, other_params = build_optimizers(model, tcfg)
    n_muon = sum(p.numel() for p in matrix_params)
    n_adamw = sum(p.numel() for p in other_params)
    logger.info(f"optimizer_split muon_tensors={len(matrix_params)} muon_params={n_muon:,} "
                f"adamw_tensors={len(other_params)} adamw_params={n_adamw:,}")

    diag = TrainingDiagnostics(model, tcfg, logger, enabled=True)

    model.train()
    for step in range(args.steps):
        tokens = torch.randint(0, mcfg.vocab_size, (args.batch_size, args.seq_len), device=device)
        targets = torch.randint(0, mcfg.vocab_size, (args.batch_size, args.seq_len), device=device)

        for opt in [muon_opt, adamw_opt]:
            opt.zero_grad(set_to_none=True)

        logits, loss = model(tokens, targets)

        if not torch.isfinite(loss):
            logger.error(f"step={step} loss_not_finite value={loss.item()}")
            return

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

        muon_opt.step()
        adamw_opt.step()

        diag.check_grad_explosion(grad_norm, step)
        diag.check_norms(step)

        logger.info(f"step={step} loss={loss.item():.4f} grad_norm={grad_norm:.4f} "
                    f"logits_shape={tuple(logits.shape)}")

    logger.info("layer_grad_summary_top_5")
    diag.log_layer_summary(args.steps, top_k=5)

    logger.info("norm_outputs_per_layer")
    for name, stats in list(diag.norm_outputs.items())[:6]:
        logger.info(f"  layer={name} std={stats['std']:.4f} mean={stats['mean']:.4f} "
                    f"abs_max={stats['abs_max']:.4f}")
    if len(diag.norm_outputs) > 6:
        logger.info(f"  ... and {len(diag.norm_outputs) - 6} more layers")

    logger.info("smoke_test_complete")


if __name__ == "__main__":
    main()
