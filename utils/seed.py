"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and all CUDA devices).

    With ``deterministic=True`` cuDNN autotuning is disabled so repeated runs
    with the same seed produce the same numbers, at some cost in speed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker its own NumPy / random seed."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)
