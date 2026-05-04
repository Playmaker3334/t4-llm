from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TrainConfig:
    micro_batch_size: int = 8
    grad_accum_steps: int = 32
    seq_len: int = 2048
    total_tokens: int = 10_000_000_000

    eval_max_batches: int = 50
    eval_interval: int = 500
    log_interval: int = 10
    checkpoint_interval: int = 1000

    # Sample generation periódico durante entrenamiento
    sample_interval: int = 100
    sample_max_new_tokens: int = 50
    sample_temperature: float = 0.8
    sample_top_k: int = 40

    # Rotación de checkpoints: mantiene los últimos N + uno cada M pasos como hito
    ckpt_keep_last: int = 5
    ckpt_milestone_every: int = 500

    muon_lr: float = 0.02
    adamw_lr: float = 3e-3
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_scale_factor: float = 0.2
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    weight_decay_matrices: float = 0.1
    weight_decay_other: float = 0.0

    warmup_pct: float = 0.015
    stable_pct: float = 0.80
    decay_pct: float = 0.185
    min_lr_ratio: float = 0.1

    grad_clip: float = 1.0

    fp16: bool = True
    grad_scaler_init: float = 65536.0

    seed: int = 42

    detect_grad_spike_factor: float = 5.0
    detect_grad_spike_window: int = 50
    detect_norm_std_min: float = 0.5
    detect_norm_std_max: float = 2.0
    detect_norm_mean_abs_max: float = 1.0
    detect_overfit_window: int = 4