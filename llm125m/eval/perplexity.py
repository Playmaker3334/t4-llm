import math
import torch
from torch.cuda.amp import autocast


@torch.no_grad()
def compute_perplexity(model, eval_loader, device, max_batches: int = 100, fp16: bool = True) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for i, (tokens, targets) in enumerate(eval_loader):
        if i >= max_batches:
            break
        tokens = tokens.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(dtype=torch.float16, enabled=fp16):
            _, loss = model(tokens, targets)
        n_tok = targets.numel()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
        n_batches += 1

    if total_tokens == 0:
        return {"loss": float("inf"), "ppl": float("inf"), "tokens": 0, "batches": 0}

    avg_loss = total_loss / total_tokens
    return {
        "loss": avg_loss,
        "ppl": math.exp(min(20.0, avg_loss)),
        "tokens": total_tokens,
        "batches": n_batches,
    }
