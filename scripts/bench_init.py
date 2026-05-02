import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model import ModelConfig
from llm125m.model import Transformer
from llm125m.model.norm import RMSNorm
from llm125m.utils.logging import setup_logger


def collect_activation_stats(model, tokens, targets):
    activations = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            def make_hook(layer_name):
                def hook(m, inp, out):
                    with torch.no_grad():
                        x = out.detach().float()
                        activations[layer_name] = {
                            "mean": x.mean().item(),
                            "std": x.std().item(),
                            "abs_max": x.abs().max().item(),
                            "abs_min": x.abs().min().item(),
                        }
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))

    logits, loss = model(tokens, targets)
    loss.backward()

    for h in hooks:
        h.remove()

    return logits, loss, activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", type=str, default="logs/bench_init.log")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    logger = setup_logger("bench_init", log_file=args.log_file)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = ModelConfig()

    logger.info(f"device={device} batch={args.batch_size} seq_len={args.seq_len} seed={args.seed}")

    model = Transformer(mcfg).to(device)
    n_total = model.num_parameters()
    n_excl = model.num_parameters(exclude_embeddings=True)
    logger.info(f"params total={n_total:,} excl_emb={n_excl:,}")

    tokens = torch.randint(0, mcfg.vocab_size, (args.batch_size, args.seq_len), device=device)
    targets = torch.randint(0, mcfg.vocab_size, (args.batch_size, args.seq_len), device=device)

    model.train()
    logits, loss, activations = collect_activation_stats(model, tokens, targets)

    expected_loss = torch.log(torch.tensor(float(mcfg.vocab_size))).item()
    delta = loss.item() - expected_loss
    logger.info(f"loss_at_init value={loss.item():.4f} expected_uniform={expected_loss:.4f} "
                f"delta={delta:+.4f}")
    if abs(delta) > 0.5:
        logger.warning(f"loss_at_init_anomaly delta={delta:+.4f} suggests_init_issue")

    logger.info(f"logits_stats mean={logits.mean().item():.4f} std={logits.std().item():.4f} "
                f"abs_max={logits.abs().max().item():.4f}")

    logger.info("rmsnorm_output_stats")
    for name, stats in activations.items():
        logger.info(f"  layer={name} std={stats['std']:.4f} mean={stats['mean']:.4f} "
                    f"abs_max={stats['abs_max']:.4f}")

    logger.info("param_grad_stats_per_module")
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        logger.info(f"  param={name} shape={tuple(p.shape)} "
                    f"grad_norm={g.norm().item():.4e} grad_std={g.std().item():.4e} "
                    f"grad_abs_max={g.abs().max().item():.4e}")

    grad_norms_per_layer = {}
    for name, p in model.named_parameters():
        if p.grad is not None and "blocks." in name:
            layer_idx = int(name.split("blocks.")[1].split(".")[0])
            grad_norms_per_layer.setdefault(layer_idx, 0.0)
            grad_norms_per_layer[layer_idx] += p.grad.norm().item() ** 2

    logger.info("grad_norm_per_block_layer")
    for idx in sorted(grad_norms_per_layer.keys()):
        logger.info(f"  block={idx} grad_norm={grad_norms_per_layer[idx] ** 0.5:.4e}")


if __name__ == "__main__":
    main()
