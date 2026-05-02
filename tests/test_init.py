import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model import ModelConfig
from llm125m.model import Transformer


def test_init_residual_projections_scaled_down():
    cfg = ModelConfig()
    cfg.n_layers = 4
    cfg.vocab_size = 256
    cfg.max_seq_len = 32
    model = Transformer(cfg)
    expected_scale = cfg.init_std / math.sqrt(2 * cfg.n_layers)
    for block in model.blocks:
        wo_std = block.attn.wo.weight.std().item()
        wd_std = block.ffn.w_down.weight.std().item()
        assert wo_std < cfg.init_std * 0.7, f"wo std {wo_std} not scaled"
        assert wd_std < cfg.init_std * 0.7, f"w_down std {wd_std} not scaled"
        assert abs(wo_std - expected_scale) < 0.01
        assert abs(wd_std - expected_scale) < 0.01


def test_init_loss_close_to_uniform():
    cfg = ModelConfig()
    cfg.n_layers = 4
    cfg.vocab_size = 1024
    cfg.max_seq_len = 64
    torch.manual_seed(42)
    model = Transformer(cfg)
    model.eval()

    tokens = torch.randint(0, cfg.vocab_size, (4, 64))
    targets = torch.randint(0, cfg.vocab_size, (4, 64))
    _, loss = model(tokens, targets)
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 0.5, \
        f"loss at init {loss.item():.4f} far from uniform {expected:.4f}"


def test_init_logits_finite():
    cfg = ModelConfig()
    cfg.n_layers = 4
    cfg.vocab_size = 1024
    cfg.max_seq_len = 64
    model = Transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, _ = model(tokens)
    assert torch.isfinite(logits).all()


def test_rmsnorm_weights_initialized_to_one():
    from llm125m.model.norm import RMSNorm
    norm = RMSNorm(64)
    assert torch.allclose(norm.weight, torch.ones(64))


def test_embedding_initialized_with_init_std():
    cfg = ModelConfig()
    cfg.n_layers = 2
    cfg.vocab_size = 8192
    cfg.max_seq_len = 32
    torch.manual_seed(0)
    model = Transformer(cfg)
    emb_std = model.embed.weight.std().item()
    assert abs(emb_std - cfg.init_std) < 0.005, f"embedding std {emb_std} != {cfg.init_std}"
