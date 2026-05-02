import torch
from collections import deque

from llm125m.model.norm import RMSNorm


class TrainingDiagnostics:
    def __init__(self, model, train_config, logger, enabled: bool = True):
        self.model = model
        self.config = train_config
        self.logger = logger
        self.enabled = enabled

        self.norm_outputs = {}
        self.hooks = []

        self.grad_norm_history = deque(maxlen=train_config.detect_grad_spike_window + 5)
        self.train_loss_history = deque(maxlen=train_config.detect_overfit_window)
        self.eval_loss_history = deque(maxlen=train_config.detect_overfit_window)

        if enabled:
            self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, RMSNorm):
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self._record_norm(n, out)
                )
                self.hooks.append(hook)

    def _record_norm(self, name, output):
        if not self.enabled:
            return
        with torch.no_grad():
            x = output.detach().float()
            self.norm_outputs[name] = {
                "mean": x.mean().item(),
                "std": x.std().item(),
                "abs_max": x.abs().max().item(),
            }

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def check_loss_health(self, loss_tensor: torch.Tensor, step: int) -> bool:
        if not torch.isfinite(loss_tensor).all():
            if self.enabled:
                self.logger.error(f"step={step} loss_nan_or_inf value={loss_tensor.item()}")
            return False
        return True

    def check_grad_explosion(self, grad_norm: float, step: int):
        self.grad_norm_history.append(grad_norm)
        window = self.config.detect_grad_spike_window
        if len(self.grad_norm_history) < window:
            return
        recent = list(self.grad_norm_history)[-window:-1]
        if not recent:
            return
        running_avg = sum(recent) / len(recent)
        if running_avg <= 0:
            return
        ratio = grad_norm / running_avg
        if ratio > self.config.detect_grad_spike_factor:
            if self.enabled:
                self.logger.warning(
                    f"step={step} grad_spike grad_norm={grad_norm:.4f} "
                    f"running_avg={running_avg:.4f} ratio={ratio:.2f}x"
                )

    def check_norms(self, step: int):
        if not self.enabled:
            return
        std_min = self.config.detect_norm_std_min
        std_max = self.config.detect_norm_std_max
        mean_max = self.config.detect_norm_mean_abs_max

        anomalies = []
        for name, stats in self.norm_outputs.items():
            std = stats["std"]
            mean = stats["mean"]
            abs_max = stats["abs_max"]
            issues = []
            if std < std_min or std > std_max:
                issues.append(f"std={std:.4f}_outside_[{std_min},{std_max}]")
            if abs(mean) > mean_max:
                issues.append(f"mean={mean:.4f}_abs_exceeds_{mean_max}")
            if not (abs_max < 1e6):
                issues.append(f"abs_max={abs_max:.2e}_extreme")
            if issues:
                anomalies.append((name, issues, std, mean, abs_max))

        for name, issues, std, mean, abs_max in anomalies:
            self.logger.warning(
                f"step={step} norm_anomaly layer={name} "
                f"std={std:.4f} mean={mean:.4f} abs_max={abs_max:.2e} "
                f"issues={','.join(issues)}"
            )

    def check_overfit(self, train_loss: float, eval_loss: float, step: int):
        self.train_loss_history.append(train_loss)
        self.eval_loss_history.append(eval_loss)

        if len(self.eval_loss_history) < self.config.detect_overfit_window:
            return

        evals = list(self.eval_loss_history)
        trains = list(self.train_loss_history)

        eval_increasing = all(evals[i] < evals[i + 1] for i in range(len(evals) - 1))
        train_decreasing = all(trains[i] > trains[i + 1] for i in range(len(trains) - 1))

        eval_delta = evals[-1] - evals[0]
        train_delta = trains[0] - trains[-1]

        if eval_increasing and train_decreasing and self.enabled:
            self.logger.warning(
                f"step={step} potential_overfit "
                f"train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} "
                f"eval_delta_window={eval_delta:.4f} train_delta_window={train_delta:.4f}"
            )

    def per_layer_grad_norms(self):
        result = {}
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                result[name] = p.grad.norm().item()
        return result

    def per_layer_param_norms(self):
        result = {}
        for name, p in self.model.named_parameters():
            result[name] = p.detach().norm().item()
        return result

    def log_step(self, step: int, loss: float, grad_norm: float, lr_factor: float,
                 tokens_per_sec: float, extra: dict = None):
        msg = (
            f"step={step} loss={loss:.4f} grad_norm={grad_norm:.4f} "
            f"lr_factor={lr_factor:.4f} tokens_per_sec={tokens_per_sec:.0f}"
        )
        if extra:
            for k, v in extra.items():
                msg += f" {k}={v}"
        self.logger.info(msg)

    def log_layer_summary(self, step: int, top_k: int = 5):
        if not self.enabled:
            return
        grad_norms = self.per_layer_grad_norms()
        if not grad_norms:
            return
        sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for name, gn in sorted_grads:
            self.logger.info(f"step={step} layer_grad_top layer={name} grad_norm={gn:.4f}")
