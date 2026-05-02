import os
import torch


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path: str, model, optimizers, scheduler, trainer, scaler=None, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model": _unwrap(model).state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "scheduler": scheduler.state_dict(),
        "global_step": trainer.global_step,
        "tokens_consumed": trainer.tokens_consumed,
        "rng_torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizers, scheduler, trainer, scaler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    _unwrap(model).load_state_dict(state["model"])
    for opt, opt_state in zip(optimizers, state["optimizers"]):
        opt.load_state_dict(opt_state)
    scheduler.load_state_dict(state["scheduler"])
    trainer.global_step = state["global_step"]
    trainer.tokens_consumed = state.get("tokens_consumed", 0)
    if "rng_torch" in state:
        torch.set_rng_state(state["rng_torch"])
    if "rng_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state.get("extra")
