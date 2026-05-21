from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AverageMeter:
    name: str
    fmt: str = ":.4f"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(state, os.path.join(output_dir, filename))
