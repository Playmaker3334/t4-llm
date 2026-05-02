import torch
import torch.nn as nn
import torch.nn.functional as F

from llm125m.model.norm import RMSNorm
from llm125m.model.rope import apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_q_heads: int, n_kv_heads: int,
                 head_dim: int, use_qk_norm: bool, norm_eps: float):
        super().__init__()
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(f"n_q_heads={n_q_heads} must be divisible by n_kv_heads={n_kv_heads}")
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.repeats = n_q_heads // n_kv_heads

        self.wq = nn.Linear(hidden_dim, n_q_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_q_heads * head_dim, hidden_dim, bias=False)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_q_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.repeats > 1:
            k = k.repeat_interleave(self.repeats, dim=1)
            v = v.repeat_interleave(self.repeats, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_q_heads * self.head_dim)
        return self.wo(out)
