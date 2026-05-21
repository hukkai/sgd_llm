from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import LlamaConfig, build_model
from utils import (
    AverageMeter,
    SOOptimizer,
    cosine_lr,
    get_param_groups,
    init_distributed,
    is_main_process,
    save_checkpoint,
    set_seed,
)


def str2bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes", "y", "on"):
        return True
    if value.lower() in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Orthogonal LLaMA-2-style pretraining")

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=9999999999)

    parser.add_argument("--orthogonal-type", type=str, default="none", choices=["none", "mlp", "atten", "all"])
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--max-position-embeddings", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--hidden-dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--tie-word-embeddings", action="store_true")

    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per rank")
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--num-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)

    parser.add_argument("--so-lr", type=float, default=1.0)
    parser.add_argument("--sub-matrix", type=int, default=16)
    parser.add_argument("--orth-beta1", type=float, default=0.95)

    parser.add_argument("--strict-stiefel-last", type=str2bool, default=True)

    return parser.parse_args()


def resolve_data_path(data_dir: str, rank: int) -> str:
    shard_path = os.path.join(data_dir, f"tokens_{rank}.bin")
    if os.path.exists(shard_path):
        return shard_path

    fallback_path = os.path.join(data_dir, "tokens_0.bin")
    if rank == 0 and os.path.exists(fallback_path):
        return fallback_path

    raise FileNotFoundError(f"Could not find token shard for rank {rank} under {data_dir}")


def build_config(args: argparse.Namespace) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        tie_word_embeddings=args.tie_word_embeddings,
    )


def create_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    exclude = ["chunk_weights"] if args.orthogonal_type != "none" else []
    param_groups = get_param_groups(model, args.weight_decay, exclude_names=exclude)
    return torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)


def load_micro_batch(
    all_tokens: np.memmap,
    micro_step: int,
    batch_size: int,
    seq_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_length = seq_length + 1
    tokens_per_micro = batch_size * sample_length
    start = micro_step * tokens_per_micro
    end = (micro_step + 1) * tokens_per_micro

    token_slice = np.asarray(all_tokens[start:end], dtype=np.int64)
    token_batch = torch.from_numpy(token_slice.reshape(batch_size, sample_length))
    token_batch = token_batch.to(device, non_blocking=True)
    return token_batch[:, :-1], token_batch[:, 1:]


def main() -> None:
    args = parse_args()
    distributed, local_rank, rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if args.global_batch_size % (args.batch_size * world_size) != 0:
        raise ValueError("global_batch_size must be divisible by batch_size * world_size")
    accum_steps = args.global_batch_size // (args.batch_size * world_size)
    if accum_steps <= 0:
        raise ValueError("Accumulation steps must be positive")
    if args.seq_length > args.max_position_embeddings:
        raise ValueError("seq_length must be <= max_position_embeddings")

    data_path = resolve_data_path(args.data_dir, rank)
    all_tokens = np.memmap(data_path, dtype=np.uint32, mode="r")

    total_micro_steps = args.num_steps * accum_steps
    tokens_per_micro = args.batch_size * (args.seq_length + 1)
    required_tokens = total_micro_steps * tokens_per_micro
    if all_tokens.shape[0] < required_tokens:
        raise ValueError(
            f"Not enough tokens in {data_path}: need {required_tokens}, found {all_tokens.shape[0]}"
        )

    config = build_config(args)
    init_chunk_weights = not distributed or rank == 0
    with torch.device(device):
        model = build_model(
            config,
            orthogonal_type=args.orthogonal_type,
            init_chunk_weights=init_chunk_weights,
        )
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = create_optimizer(args, model)
    module = model.module if hasattr(model, "module") else model
    orth_opt = None
    if args.orthogonal_type != "none":
        orth_opt = SOOptimizer(
            module.chunk_weights,
            lr=args.lr * args.so_lr,
            beta1=args.orth_beta1,
            sub_matrix=args.sub_matrix,
            strict_stiefel=args.strict_stiefel_last,
        )

    optimizer.zero_grad(set_to_none=True)
    loss_meter = AverageMeter("loss")
    start_time = time.time()
    start_micro_step = 0

    warmup_steps = int(args.num_steps * 0.01)
    for micro_step in range(start_micro_step, total_micro_steps):
        step = micro_step // accum_steps
        lr = cosine_lr(step, args.num_steps, warmup_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_ids, labels = load_micro_batch(all_tokens, micro_step, args.batch_size, args.seq_length, device)
        should_sync = (micro_step + 1) % accum_steps == 0
        sync_context = (
            model.no_sync() if distributed and hasattr(model, "no_sync") and not should_sync else nullcontext()
        )
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )

        with sync_context:
            with autocast_ctx:
                output = model(input_ids=input_ids, labels=labels)
                loss = output["loss"]
            (loss / accum_steps).backward()

        loss_meter.update(loss.item(), input_ids.size(0))

        if not should_sync:
            continue

        if args.clip_grad and args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        # strict stiefel 50 times in the entire training
        strict_stiefel_steps = args.num_steps // 50 or 1
        is_last_step = (step + 1) % strict_stiefel_steps == 0

        if orth_opt is not None:
            orth_opt.step(lr=lr * args.so_lr, is_last=is_last_step)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        completed_step = step + 1
        if is_main_process() and (
            completed_step % args.log_interval == 0 or completed_step == 1 or is_last_step
        ):
            elapsed = max(time.time() - start_time, 1e-6)
            tokens_seen = completed_step * args.global_batch_size * args.seq_length
            tokens_per_second = tokens_seen / elapsed
            print(
                f"Step {completed_step:06d}/{args.num_steps:06d} "
                f"LR {lr:.6e} Loss {loss_meter.avg:.4f} Tokens/s {tokens_per_second:.1f}"
            )
            loss_meter.reset()

        if is_main_process() and (
            completed_step % args.save_freq == 0 or completed_step == args.num_steps
        ):
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "step": completed_step,
                    "config": vars(config),
                    "args": vars(args),
                },
                args.output,
                filename=f"checkpoint_{completed_step:06d}.pth",
            )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
