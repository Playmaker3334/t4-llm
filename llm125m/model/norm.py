import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = (x_fp32 * rms).to(orig_dtype)
        return out * self.weight
