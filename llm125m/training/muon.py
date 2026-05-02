import torch
from torch.optim import Optimizer


@torch.no_grad()
def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if G.ndim != 2:
        raise ValueError(f"newton_schulz expects 2D tensor, got {G.ndim}D")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16) if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    norm = X.norm() + eps
    X = X / norm
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.0, ns_steps: int = 5,
                 scale_factor: float = 0.2):
        if lr < 0:
            raise ValueError(f"invalid lr {lr}")
        if not 0 <= momentum < 1:
            raise ValueError(f"invalid momentum {momentum}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, scale_factor=scale_factor)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            sf = group["scale_factor"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(f"Muon requires 2D parameters, got shape {tuple(p.shape)}")

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                nesterov_grad = grad.add(buf, alpha=momentum)

                ortho = newton_schulz_orthogonalize(nesterov_grad, steps=ns_steps)

                d0, d1 = p.shape
                scale = sf * (max(d0, d1) ** 0.5)

                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(ortho, alpha=-lr * scale)

        return loss
