"""Shape-faithful synthetic samples for every task.

They allow the whole pipeline (model, training loop, analysis tools) to be
exercised without the benchmark datasets, which are not distributed with the
code.  The tensors have the same layout as the benchmark caches:

* fer : ``[T, N, Sz, Sz]`` optical-flow magnitude patches (non-negative)
* esd : ``[T, N, n_w]`` z-scored window signals
* har : ``[T, N, 3]`` joint coordinates or bone displacements
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from models.unified_stgl import sequence_length


def sample_shape(cfg) -> Tuple[int, ...]:
    """Shape of one sample for the configured task (without the batch axis)."""
    task = cfg.data.task
    length = sequence_length(cfg)
    nodes = int(cfg.data.num_nodes)
    if task == "fer":
        return (length, nodes, int(cfg.data.patch_size), int(cfg.data.patch_size))
    if task == "esd":
        return (length, nodes, int(cfg.data.window_samples))
    if task == "har":
        return (length, nodes, 3)
    raise ValueError(f"Unknown task '{task}'")


def make_sample(cfg, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """One random sample with a label and, for facial data, landmark positions."""
    task = cfg.data.task
    shape = sample_shape(cfg)
    if task == "fer":
        x = rng.gamma(shape=2.0, scale=0.5, size=shape).astype(np.float32)
        landmarks = rng.normal(size=(shape[0] + 1, shape[1], 2)).astype(np.float32).cumsum(axis=0)
    elif task == "esd":
        x = rng.normal(size=shape).astype(np.float32)
        landmarks = None
    else:
        x = rng.normal(scale=0.3, size=shape).astype(np.float32).cumsum(axis=0) * 0.1
        landmarks = x[..., :2]
    sample = {"x": x, "y": np.int64(rng.integers(int(cfg.data.num_classes)))}
    if landmarks is not None:
        sample["landmarks"] = landmarks
    return sample


class SyntheticDataset(Dataset):
    """Random dataset producing ``num_samples`` samples for the configured task."""

    def __init__(self, cfg, num_samples: int = 16, seed: int = 0) -> None:
        self.cfg = cfg
        self.num_samples = num_samples
        rng = np.random.default_rng(seed)
        self.samples = [make_sample(cfg, rng) for _ in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        out = {"x": torch.from_numpy(sample["x"]), "y": torch.tensor(sample["y"])}
        if "landmarks" in sample:
            out["landmarks"] = torch.from_numpy(sample["landmarks"])
        return out


def make_batch(cfg, batch_size: int = 2, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience: a single ``(inputs, labels)`` batch for quick checks."""
    dataset = SyntheticDataset(cfg, batch_size, seed)
    x = torch.stack([dataset[i]["x"] for i in range(batch_size)])
    y = torch.stack([dataset[i]["y"] for i in range(batch_size)])
    return x, y
