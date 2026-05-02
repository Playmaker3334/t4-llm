import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm125m.model.rope import precompute_rope_cache, apply_rope


def test_rope_cache_shapes():
    cos, sin = precompute_rope_cache(head_dim=64, max_seq_len=128, theta=10000.0,
                                     device="cpu", dtype=torch.float32)
    assert cos.shape == (128, 32)
    assert sin.shape == (128, 32)


def test_apply_rope_preserves_norm_per_head():
    torch.manual_seed(0)
    head_dim = 64
    seq_len = 16
    x = torch.randn(2, seq_len, 4, head_dim)
    cos, sin = precompute_rope_cache(head_dim, seq_len, 10000.0, "cpu", torch.float32)
    y = apply_rope(x, cos[:seq_len], sin[:seq_len])
    norm_x = x.norm(dim=-1)
    norm_y = y.norm(dim=-1)
    assert torch.allclose(norm_x, norm_y, atol=1e-4)


def test_apply_rope_position_zero_is_identity():
    head_dim = 32
    x = torch.randn(1, 1, 1, head_dim)
    cos, sin = precompute_rope_cache(head_dim, 4, 10000.0, "cpu", torch.float32)
    y = apply_rope(x, cos[:1], sin[:1])
    assert torch.allclose(x, y, atol=1e-5)


def test_rope_dtype_preserved():
    head_dim = 32
    x = torch.randn(1, 4, 1, head_dim, dtype=torch.float16)
    cos, sin = precompute_rope_cache(head_dim, 4, 10000.0, "cpu", torch.float16)
    y = apply_rope(x, cos[:4], sin[:4])
    assert y.dtype == torch.float16
