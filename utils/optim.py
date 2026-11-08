"""Optimiser and learning-rate schedule factories.

* ``adam``  Adam with weight decay (facial and EEG experiments).
* ``sgd``   SGD with Nesterov momentum (skeleton experiments).
* ``cosine`` cosine annealing over the training epochs.
* ``step``   linear warm-up for ``warmup_epochs`` followed by step decay by
             ``gamma`` at every milestone.

Schedulers step once per epoch.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(cfg_train, parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    name = cfg_train.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, lr=float(cfg_train.lr), weight_decay=float(cfg_train.weight_decay))
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(cfg_train.lr),
            momentum=float(cfg_train.momentum),
            weight_decay=float(cfg_train.weight_decay),
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer '{cfg_train.optimizer}'")


def build_scheduler(cfg_train, optimizer: torch.optim.Optimizer) -> LambdaLR:
    name = cfg_train.scheduler.lower()
    epochs = int(cfg_train.epochs)
    warmup = int(cfg_train.get("warmup_epochs", 0))
    milestones = sorted(int(m) for m in cfg_train.get("milestones", []))
    gamma = float(cfg_train.get("gamma", 0.1))

    def warmup_factor(epoch: int) -> float:
        return (epoch + 1) / warmup if warmup > 0 and epoch < warmup else 1.0

    if name == "cosine":
        def factor(epoch: int) -> float:
            return warmup_factor(epoch) * 0.5 * (1.0 + math.cos(math.pi * min(epoch, epochs) / epochs))
    elif name == "step":
        def factor(epoch: int) -> float:
            return warmup_factor(epoch) * gamma ** sum(epoch >= m for m in milestones)
    elif name == "none":
        def factor(epoch: int) -> float:
            return warmup_factor(epoch)
    else:
        raise ValueError(f"Unknown scheduler '{cfg_train.scheduler}'")
    return LambdaLR(optimizer, factor)
