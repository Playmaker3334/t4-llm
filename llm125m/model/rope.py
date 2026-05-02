import torch


def precompute_rope_cache(head_dim: int, max_seq_len: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    cos_full = torch.cat((cos, cos), dim=-1)
    sin_full = torch.cat((sin, sin), dim=-1)
    if x.dim() == 4 and x.shape[1] == cos_full.shape[0]:
        cos_full = cos_full.unsqueeze(0).unsqueeze(2)
        sin_full = sin_full.unsqueeze(0).unsqueeze(2)
    elif x.dim() == 4 and x.shape[2] == cos_full.shape[0]:
        cos_full = cos_full.unsqueeze(0).unsqueeze(0)
        sin_full = sin_full.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3 and x.shape[1] == cos_full.shape[0]:
        cos_full = cos_full.unsqueeze(0)
        sin_full = sin_full.unsqueeze(0)
    return (x * cos_full + rotated * sin_full).to(x.dtype)
