import math


class WSDScheduler:
    def __init__(self, optimizers, base_lrs, total_steps,
                 warmup_pct, stable_pct, decay_pct, min_lr_ratio):
        self.optimizers = optimizers
        self.base_lrs = list(base_lrs)
        self.total_steps = total_steps
        self.warmup_steps = max(1, int(warmup_pct * total_steps))
        self.stable_steps = int(stable_pct * total_steps)
        self.decay_steps = max(1, total_steps - self.warmup_steps - self.stable_steps)
        self.min_lr_ratio = min_lr_ratio
        self.step_count = 0

    def get_lr_factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        if step < self.warmup_steps + self.stable_steps:
            return 1.0
        decay_step = step - self.warmup_steps - self.stable_steps
        progress = min(1.0, decay_step / self.decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self) -> float:
        factor = self.get_lr_factor(self.step_count)
        for opt, base_lr in zip(self.optimizers, self.base_lrs):
            for group in opt.param_groups:
                group["lr"] = base_lr * factor
        self.step_count += 1
        return factor

    def current_factor(self) -> float:
        return self.get_lr_factor(max(0, self.step_count - 1))

    def state_dict(self):
        return {"step_count": self.step_count}

    def load_state_dict(self, state):
        self.step_count = state["step_count"]
