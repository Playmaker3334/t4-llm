import time


T4_PEAK_FLOPS_FP16 = 65.0e12


class ThroughputMeter:
    def __init__(self, model_params: int, hidden_dim: int, n_layers: int):
        self.start = time.time()
        self.tokens_processed = 0
        self.steps_processed = 0
        self.model_params = model_params
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def update(self, tokens: int, steps: int = 1):
        self.tokens_processed += tokens
        self.steps_processed += steps

    def reset(self):
        self.start = time.time()
        self.tokens_processed = 0
        self.steps_processed = 0

    def elapsed(self) -> float:
        return time.time() - self.start

    def tokens_per_second(self) -> float:
        return self.tokens_processed / max(1e-9, self.elapsed())

    def steps_per_second(self) -> float:
        return self.steps_processed / max(1e-9, self.elapsed())

    def estimated_flops_per_token(self, seq_len: int) -> float:
        return 6.0 * self.model_params + 12.0 * self.n_layers * self.hidden_dim * seq_len

    def estimated_mfu(self, seq_len: int, peak_flops: float = T4_PEAK_FLOPS_FP16) -> float:
        flops_per_token = self.estimated_flops_per_token(seq_len)
        achieved = flops_per_token * self.tokens_per_second()
        return achieved / peak_flops
