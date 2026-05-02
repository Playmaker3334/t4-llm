import torch
import torch.nn as nn

from llm125m.model.norm import RMSNorm
from llm125m.model.attention import GroupedQueryAttention
from llm125m.model.ffn import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm_attn = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(
            hidden_dim=config.hidden_dim,
            n_q_heads=config.n_q_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            use_qk_norm=config.use_qk_norm,
            norm_eps=config.norm_eps,
        )
        self.norm_ffn = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.ffn = SwiGLU(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.ffn_intermediate,
            clamp_value=config.swiglu_clamp,
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        x = x + self.ffn(self.norm_ffn(x))
        return x
