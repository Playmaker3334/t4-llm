import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model import ModelConfig
from llm125m.model import Transformer


def test_param_count_around_125m():
    cfg = ModelConfig()
    model = Transformer(cfg)
    n = model.num_parameters()
    assert 120_000_000 < n < 130_000_000, f"expected ~125M params, got {n:,}"


def test_forward_output_shape():
    cfg = ModelConfig()
    cfg.max_seq_len = 64
    cfg.vocab_size = 1024
    cfg.n_layers = 2
    model = Transformer(cfg)
    model.eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(tokens)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss():
    cfg = ModelConfig()
    cfg.max_seq_len = 64
    cfg.vocab_size = 1024
    cfg.n_layers = 2
    model = Transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    _, loss = model(tokens, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_tied_embeddings_share_weights():
    cfg = ModelConfig()
    cfg.tie_embeddings = True
    cfg.n_layers = 2
    cfg.vocab_size = 1024
    cfg.max_seq_len = 64
    model = Transformer(cfg)
    assert not hasattr(model, "lm_head")


def test_rejects_seq_longer_than_max():
    cfg = ModelConfig()
    cfg.max_seq_len = 32
    cfg.vocab_size = 256
    cfg.n_layers = 2
    model = Transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 64))
    try:
        model(tokens)
    except ValueError:
        return
    assert False, "expected ValueError for seq_len > max_seq_len"
