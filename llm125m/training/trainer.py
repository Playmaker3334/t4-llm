import math
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from llm125m.utils.diagnostics import TrainingDiagnostics
from llm125m.utils.throughput import ThroughputMeter
from llm125m.training.ddp import all_reduce_mean


class Trainer:
    def __init__(self, model, optimizers, scheduler, train_loader, eval_loader,
                 train_config, model_config, logger, device, world_size: int = 1, rank: int = 0):
        self.model = model
        self.optimizers = optimizers
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.tcfg = train_config
        self.mcfg = model_config
        self.logger = logger
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.is_main = (rank == 0)

        unwrapped = self._unwrap()
        self.diag = TrainingDiagnostics(unwrapped, train_config, logger, enabled=self.is_main)
        self.scaler = GradScaler(enabled=train_config.fp16, init_scale=train_config.grad_scaler_init)
        self.throughput = ThroughputMeter(
            model_params=sum(p.numel() for p in unwrapped.parameters()),
            hidden_dim=model_config.hidden_dim,
            n_layers=model_config.n_layers,
        )

        self.global_step = 0
        self.tokens_consumed = 0

    def _unwrap(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def train_step(self, batch_iter):
        self._zero_grad()
        accum_loss = 0.0
        n_micro = 0

        for _ in range(self.tcfg.grad_accum_steps):
            try:
                tokens, targets = next(batch_iter)
            except StopIteration:
                return None, None, None, None

            tokens = tokens.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast(dtype=torch.float16, enabled=self.tcfg.fp16):
                _, loss = self.model(tokens, targets)
                loss = loss / self.tcfg.grad_accum_steps

            self.scaler.scale(loss).backward()
            accum_loss += loss.item() * self.tcfg.grad_accum_steps
            n_micro += 1

        for opt in self.optimizers:
            self.scaler.unscale_(opt)

        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)

        for opt in self.optimizers:
            self.scaler.step(opt)
        self.scaler.update()

        lr_factor = self.scheduler.step()

        avg_loss = accum_loss / n_micro
        return avg_loss, grad_norm.item(), lr_factor, n_micro

    @torch.no_grad()
    def evaluate(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for i, (tokens, targets) in enumerate(self.eval_loader):
            if i >= self.tcfg.eval_max_batches:
                break
            tokens = tokens.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with autocast(dtype=torch.float16, enabled=self.tcfg.fp16):
                _, loss = self.model(tokens, targets)
            n_tok = targets.numel()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok
        self.model.train()

        if total_tokens == 0:
            return float("inf"), float("inf")

        avg_loss = total_loss / total_tokens
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        loss_tensor = all_reduce_mean(loss_tensor)
        avg_loss = loss_tensor.item()
        return avg_loss, math.exp(min(20.0, avg_loss))

    def fit(self, total_steps: int, ckpt_callback=None):
        self.model.train()
        loader_iter = iter(self.train_loader)

        if self.is_main:
            self.logger.info(
                f"fit_start total_steps={total_steps} start_step={self.global_step} "
                f"tokens_consumed={self.tokens_consumed}"
            )

        while self.global_step < total_steps:
            t0 = time.time()
            result = self.train_step(loader_iter)

            if result[0] is None:
                loader_iter = iter(self.train_loader)
                continue

            loss, grad_norm, lr_factor, n_micro = result

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_time = time.time() - t0

            tokens_this_step = (
                self.tcfg.micro_batch_size * n_micro * self.tcfg.seq_len * self.world_size
            )
            self.tokens_consumed += tokens_this_step
            self.throughput.update(tokens_this_step, 1)

            loss_tensor = torch.tensor(loss, device=self.device)
            if not self.diag.check_loss_health(loss_tensor, self.global_step):
                if self.is_main:
                    self.logger.error(f"step={self.global_step} aborting_due_to_loss_anomaly")
                break

            self.diag.check_grad_explosion(grad_norm, self.global_step)

            if self.is_main and self.global_step % self.tcfg.log_interval == 0:
                tps = tokens_this_step / max(1e-6, step_time)
                self.diag.log_step(
                    step=self.global_step,
                    loss=loss,
                    grad_norm=grad_norm,
                    lr_factor=lr_factor,
                    tokens_per_sec=tps,
                    extra={
                        "scaler": f"{self.scaler.get_scale():.0f}",
                        "tokens_total": self.tokens_consumed,
                        "step_time_s": f"{step_time:.3f}",
                    },
                )
                self.diag.check_norms(self.global_step)

            if self.global_step > 0 and self.global_step % self.tcfg.eval_interval == 0:
                eval_loss, ppl = self.evaluate()
                if self.is_main:
                    self.logger.info(
                        f"step={self.global_step} eval_loss={eval_loss:.4f} eval_ppl={ppl:.2f}"
                    )
                    self.diag.check_overfit(loss, eval_loss, self.global_step)

            if (ckpt_callback is not None and self.is_main
                    and self.global_step > 0
                    and self.global_step % self.tcfg.checkpoint_interval == 0):
                ckpt_callback(self.global_step)

            self.global_step += 1

        if self.is_main:
            self.logger.info(
                f"fit_end final_step={self.global_step} tokens_consumed={self.tokens_consumed}"
            )
