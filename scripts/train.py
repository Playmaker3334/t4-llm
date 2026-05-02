import argparse
import os
import sys

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model import ModelConfig
from configs.training import TrainConfig
from configs.data import DataConfig
from llm125m.model import Transformer
from llm125m.training import (
    build_optimizers, WSDScheduler, Trainer,
    save_checkpoint, load_checkpoint, setup_ddp, cleanup_ddp, is_main_process,
)
from llm125m.data import build_train_loader, build_eval_loader
from llm125m.utils import setup_logger, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--shard-dir", type=str, default=None)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"train_rank{rank}.log")
    logger = setup_logger("train", log_file=log_path, rank=rank)

    mcfg = ModelConfig()
    tcfg = TrainConfig()
    dcfg = DataConfig()
    if args.shard_dir is not None:
        dcfg.shard_dir = args.shard_dir

    set_seed(tcfg.seed + rank)

    if is_main_process(rank):
        logger.info(f"config_model layers={mcfg.n_layers} hidden={mcfg.hidden_dim} "
                    f"vocab={mcfg.vocab_size} max_seq={mcfg.max_seq_len} "
                    f"qk_norm={mcfg.use_qk_norm} swiglu_clamp={mcfg.swiglu_clamp}")
        logger.info(f"config_train micro_batch={tcfg.micro_batch_size} "
                    f"grad_accum={tcfg.grad_accum_steps} seq_len={tcfg.seq_len} "
                    f"total_tokens={tcfg.total_tokens} fp16={tcfg.fp16}")
        logger.info(f"distributed world_size={world_size} local_rank={local_rank} device={device}")

    model = Transformer(mcfg).to(device)
    n_total = model.num_parameters()
    n_excl_emb = model.num_parameters(exclude_embeddings=True)
    if is_main_process(rank):
        logger.info(f"model_built params_total={n_total:,} params_excl_emb={n_excl_emb:,}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True,
                    find_unused_parameters=False)

    target_for_optimizer = model.module if world_size > 1 else model
    muon_opt, adamw_opt, matrix_params, other_params = build_optimizers(target_for_optimizer, tcfg)

    n_muon = sum(p.numel() for p in matrix_params)
    n_adamw = sum(p.numel() for p in other_params)
    if is_main_process(rank):
        logger.info(f"optimizer_split muon_params={n_muon:,} muon_tensors={len(matrix_params)} "
                    f"adamw_params={n_adamw:,} adamw_tensors={len(other_params)}")
        logger.info(f"optimizer_lrs muon_lr={tcfg.muon_lr} adamw_lr={tcfg.adamw_lr} "
                    f"muon_wd={tcfg.weight_decay_matrices} adamw_wd={tcfg.weight_decay_other}")

    train_loader = build_train_loader(dcfg, tcfg)
    eval_loader = build_eval_loader(dcfg, tcfg)

    tokens_per_step = tcfg.micro_batch_size * tcfg.grad_accum_steps * tcfg.seq_len * world_size
    total_steps = tcfg.total_tokens // tokens_per_step

    scheduler = WSDScheduler(
        optimizers=[muon_opt, adamw_opt],
        base_lrs=[tcfg.muon_lr, tcfg.adamw_lr],
        total_steps=total_steps,
        warmup_pct=tcfg.warmup_pct,
        stable_pct=tcfg.stable_pct,
        decay_pct=tcfg.decay_pct,
        min_lr_ratio=tcfg.min_lr_ratio,
    )

    if is_main_process(rank):
        logger.info(f"schedule total_steps={total_steps} tokens_per_step={tokens_per_step:,} "
                    f"warmup={scheduler.warmup_steps} stable={scheduler.stable_steps} "
                    f"decay={scheduler.decay_steps}")

    trainer = Trainer(
        model=model,
        optimizers=[muon_opt, adamw_opt],
        scheduler=scheduler,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_config=tcfg,
        model_config=mcfg,
        logger=logger,
        device=device,
        world_size=world_size,
        rank=rank,
    )

    if args.resume is not None:
        load_checkpoint(args.resume, model, [muon_opt, adamw_opt], scheduler, trainer,
                        scaler=trainer.scaler, map_location=device)
        if is_main_process(rank):
            logger.info(f"checkpoint_loaded path={args.resume} step={trainer.global_step} "
                        f"tokens_consumed={trainer.tokens_consumed}")

    def ckpt_cb(step):
        path = os.path.join(args.ckpt_dir, f"ckpt_step{step:08d}.pt")
        save_checkpoint(path, model, [muon_opt, adamw_opt], scheduler, trainer,
                        scaler=trainer.scaler)
        logger.info(f"checkpoint_saved path={path} step={step} "
                    f"tokens_consumed={trainer.tokens_consumed}")

    try:
        trainer.fit(total_steps, ckpt_callback=ckpt_cb)
    finally:
        if is_main_process(rank):
            final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
            save_checkpoint(final_path, model, [muon_opt, adamw_opt], scheduler, trainer,
                            scaler=trainer.scaler)
            logger.info(f"checkpoint_saved_final path={final_path}")
        cleanup_ddp(world_size)


if __name__ == "__main__":
    main()
