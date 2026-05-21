from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, local_rank, dist.get_rank(), dist.get_world_size()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0
