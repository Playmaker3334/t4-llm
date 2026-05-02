import torch
import torch.nn as nn
import torch.nn.functional as F

from llm125m.model.norm import RMSNorm
from llm125m.model.block import TransformerBlock
from llm125m.model.rope import precompute_rope_cache
from llm125m.model.init_weights import init_weights


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layers)
        ])
        self.norm_final = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        init_weights(self, config)

        cos, sin = precompute_rope_cache(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
            device="cpu",
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        B, T = tokens.shape
        if T > self.config.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.config.max_seq_len}")

        x = self.embed(tokens)
        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm_final(x)

        if self.config.tie_embeddings:
            logits = x @ self.embed.weight.T
        else:
            logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed.weight.numel()
        return n

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None, top_p: float = None,
                 eos_token_id: int = None):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = tokens[:, -self.config.max_seq_len:]
            logits, _ = self.forward(ctx)
            logits = logits[:, -1, :] / max(1e-8, temperature)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative = probs.cumsum(dim=-1)
                mask = cumulative > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return tokens
