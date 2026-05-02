from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_dim: int = 576
    n_layers: int = 30
    n_q_heads: int = 9
    n_kv_heads: int = 3
    head_dim: int = 64
    ffn_intermediate: int = 1536
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    swiglu_clamp: float = 10.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02
    use_qk_norm: bool = True
