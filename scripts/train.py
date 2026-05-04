import argparse
import glob
import os
import re
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


# Prompts fijos para sample generation periódico.
SAMPLE_PROMPTS = [
    "The history of artificial intelligence",
    "In the field of physics,",
    "Once upon a time, in a small village",
]


def find_latest_checkpoint(ckpt_dir: str):

    latest_pointer = os.path.join(ckpt_dir, 'latest.txt')
    if os.path.isfile(latest_pointer):
        with open(latest_pointer) as f:
            path = f.read().strip()
        if os.path.isfile(path):
            return path
    
    # Fallback: buscar por patrón
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_step*.pt')))
    return candidates[-1] if candidates else None


def step_from_filename(path: str):
    """Extrae el número de paso del nombre de archivo."""
    m = re.search(r'ckpt_step(\d+)\.pt$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def rotate_checkpoints(ckpt_dir: str, keep_last: int, milestone_every: int, logger):
    """Mantiene los últimos `keep_last` checkpoints más uno cada `milestone_every` pasos
    como hitos permanentes. Borra el resto.
    """
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_step*.pt')))
    if len(candidates) <= keep_last:
        return
    
    # Identificar qué conservar
    by_step = [(step_from_filename(p), p) for p in candidates]
    by_step = [(s, p) for s, p in by_step if s >= 0]
    by_step.sort(key=lambda x: x[0])
    
    last_n_paths = {p for _, p in by_step[-keep_last:]}
    milestone_paths = {p for s, p in by_step if s % milestone_every == 0 and s > 0}
    keep = last_n_paths | milestone_paths
    
    removed = 0
    for _, p in by_step:
        if p not in keep:
            try:
                os.remove(p)
                removed += 1
            except OSError as e:
                logger.warning(f"failed_to_remove_ckpt path={p} error={e}")
    
    if removed > 0:
        logger.info(f"ckpt_rotation removed={removed} kept={len(keep)}")


def write_latest_pointer(ckpt_dir: str, path: str):
    """Escribe latest.txt apuntando al checkpoint más reciente."""
    with open(os.path.join(ckpt_dir, 'latest.txt'), 'w') as f:
        f.write(path)


def build_sample_callback(trainer, tokenizer_path: str, logger, tcfg):
    """Construye un callback que genera texto desde prompts fijos y lo loguea."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning("transformers no disponible; sample_callback deshabilitado")
        return None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        logger.warning(f"no se pudo cargar tokenizer={tokenizer_path}: {e}")
        return None
    
    # Tokenizar los prompts una sola vez al inicio
    prompts_encoded = []
    for prompt_text in SAMPLE_PROMPTS:
        ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        prompts_encoded.append((prompt_text, torch.tensor(ids, dtype=torch.long)))
    
    logger.info(f"sample_callback_ready tokenizer={tokenizer_path} n_prompts={len(prompts_encoded)}")
    
    def callback(step):
        for i, (prompt_text, prompt_ids) in enumerate(prompts_encoded):
            result = trainer.sample_generation(
                prompt_ids,
                max_new_tokens=tcfg.sample_max_new_tokens,
                temperature=tcfg.sample_temperature,
                top_k=tcfg.sample_top_k,
            )
            if result is None:
                continue
            
            generated_text = tokenizer.decode(result['generated_ids'], skip_special_tokens=True)
            # Sanitizar saltos de línea en el output para que el log quede en una línea
            generated_text_safe = generated_text.replace('\n', ' \\n ').replace('\r', '')
            
            logger.info(
                f"sample step={step} prompt_id={i} "
                f"prompt={prompt_text!r} "
                f"output={generated_text_safe!r} "
                f"n_tokens={result['n_generated']} "
                f"tps={result['tokens_per_sec']:.1f}"
            )
    
    return callback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path explícito al checkpoint. Si se omite, intenta auto-resume desde ckpt-dir.")
    parser.add_argument("--no-auto-resume", action="store_true",
                        help="Deshabilita el auto-resume desde ckpt-dir.")
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
        logger.info(f"config_intervals ckpt={tcfg.checkpoint_interval} "
                    f"eval={tcfg.eval_interval} sample={tcfg.sample_interval} "
                    f"log={tcfg.log_interval}")
        logger.info(f"config_ckpt_rotation keep_last={tcfg.ckpt_keep_last} "
                    f"milestone_every={tcfg.ckpt_milestone_every}")
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

    resume_path = args.resume
    if resume_path is None and not args.no_auto_resume:
        resume_path = find_latest_checkpoint(args.ckpt_dir)
        if resume_path is not None and is_main_process(rank):
            logger.info(f"auto_resume_detected path={resume_path}")
    
    if resume_path is not None:
        load_checkpoint(resume_path, model, [muon_opt, adamw_opt], scheduler, trainer,
                        scaler=trainer.scaler, map_location=device)
        if is_main_process(rank):
            logger.info(f"checkpoint_loaded path={resume_path} step={trainer.global_step} "
                        f"tokens_consumed={trainer.tokens_consumed}")
    else:
        if is_main_process(rank):
            logger.info("starting_fresh no_checkpoint_found")

    def ckpt_cb(step):
        path = os.path.join(args.ckpt_dir, f"ckpt_step{step:08d}.pt")
        save_checkpoint(path, model, [muon_opt, adamw_opt], scheduler, trainer,
                        scaler=trainer.scaler)
        write_latest_pointer(args.ckpt_dir, path)
        rotate_checkpoints(args.ckpt_dir, tcfg.ckpt_keep_last,
                           tcfg.ckpt_milestone_every, logger)
        logger.info(f"checkpoint_saved path={path} step={step} "
                    f"tokens_consumed={trainer.tokens_consumed}")

    sample_cb = None
    if is_main_process(rank):
        sample_cb = build_sample_callback(trainer, dcfg.tokenizer_path, logger, tcfg)


    try:
        trainer.fit(total_steps, ckpt_callback=ckpt_cb, sample_callback=sample_cb)
    finally:
        if is_main_process(rank):
            final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
            save_checkpoint(final_path, model, [muon_opt, adamw_opt], scheduler, trainer,
                            scaler=trainer.scaler)
            write_latest_pointer(args.ckpt_dir, final_path)
            logger.info(f"checkpoint_saved_final path={final_path}")
        cleanup_ddp(world_size)


if __name__ == "__main__":
    main()