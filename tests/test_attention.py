import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm125m.model.attention import GroupedQueryAttention
from llm125m.model.rope import precompute_rope_cache


def test_gqa_output_shape():
    hidden_dim = 576
    n_q, n_kv, head_dim = 9, 3, 64
    attn = GroupedQueryAttention(hidden_dim, n_q, n_kv, head_dim, use_qk_norm=True, norm_eps=1e-5)
    cos, sin = precompute_rope_cache(head_dim, 64, 10000.0, "cpu", torch.float32)
    x = torch.randn(2, 32, hidden_dim)
    out = attn(x, cos[:32], sin[:32])
    assert out.shape == (2, 32, hidden_dim)


def test_gqa_repeats_kv_correctly():
    hidden_dim = 64
    n_q, n_kv, head_dim = 4, 2, 16
    attn = GroupedQueryAttention(hidden_dim, n_q, n_kv, head_dim, use_qk_norm=False, norm_eps=1e-5)
    assert attn.repeats == 2


def test_gqa_rejects_invalid_head_split():
    try:
        GroupedQueryAttention(576, n_q_heads=9, n_kv_heads=4, head_dim=64,
                              use_qk_norm=False, norm_eps=1e-5)
    except ValueError:
        return
    assert False, "expected ValueError when n_q_heads not divisible by n_kv_heads"


def test_qk_norm_modules_present_when_enabled():
    attn = GroupedQueryAttention(576, 9, 3, 64, use_qk_norm=True, norm_eps=1e-5)
    assert hasattr(attn, "q_norm")
    assert hasattr(attn, "k_norm")


def test_qk_norm_modules_absent_when_disabled():
    attn = GroupedQueryAttention(576, 9, 3, 64, use_qk_norm=False, norm_eps=1e-5)
    assert not hasattr(attn, "q_norm")
    assert not hasattr(attn, "k_norm")


def test_attention_is_finite_for_random_input():
    attn = GroupedQueryAttention(576, 9, 3, 64, use_qk_norm=True, norm_eps=1e-5)
    cos, sin = precompute_rope_cache(64, 128, 10000.0, "cpu", torch.float32)
    x = torch.randn(2, 64, 576)
    out = attn(x, cos[:64], sin[:64])
    assert torch.isfinite(out).all()
