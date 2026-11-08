"""Common dataset interface and batch helpers.

Every dataset used with ``main.py`` returns dictionaries with

    x         : task-specific input tensor with leading ``[T, N, ...]`` axes
    y         : class index (``int64``)
    subject   : subject / patient identifier used by subject-independent splits
    split     : ``train`` / ``test`` for datasets with an official partition
    landmarks : (facial data) aligned landmark positions ``[T + 1, N, 2]``

``collate`` stacks the tensors and keeps the string fields as lists, and
``transition_prior`` derives the input-level transition scores needed by the
``flow_mag`` and ``landmark_vel`` ablations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class GraphSequenceDataset(Dataset):
    """Base class holding a list of sample records.

    Subclasses implement ``load(record)``; augmentation is attached per
    training fold through ``SubsetWithTransform`` so that test samples are
    never transformed.
    """

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self.records = records
        self.train_transform = None

    def __len__(self) -> int:
        return len(self.records)

    def load(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Return the sample dictionary for a record (subclasses override)."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.load(self.records[index])

    def training_indices(self, indices: np.ndarray, seed: int = 0) -> np.ndarray:
        """Hook for datasets that re-balance the training fold (identity by default)."""
        return indices

    @property
    def subjects(self) -> List[str]:
        return [str(r.get("subject", "")) for r in self.records]

    @property
    def splits(self) -> List[str]:
        return [str(r.get("split", "")) for r in self.records]

    @property
    def labels(self) -> List[int]:
        return [int(r["y"]) for r in self.records]


class SubsetWithTransform(Dataset):
    """A subset of a dataset with an optional transform applied on access."""

    def __init__(self, dataset: Dataset, indices, transform=None) -> None:
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.dataset[self.indices[index]]
        return self.transform(sample) if self.transform is not None else sample


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensor fields, keep other fields as lists."""
    out: Dict[str, Any] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            out[key] = torch.tensor(values)
        else:
            out[key] = values
    return out


def transition_prior(batch: Dict[str, Any], measure: str, task: str) -> Optional[torch.Tensor]:
    """Input-level transition scores ``[B, T, N]`` in [0, 1] for the ablation measures.

    * ``flow_mag``     mean optical-flow magnitude of each landmark patch (facial data)
    * ``landmark_vel`` displacement of each node between consecutive positions
                       (facial landmarks or skeleton joints)
    """
    if measure not in ("flow_mag", "landmark_vel"):
        return None
    x = batch["x"]
    if measure == "flow_mag":
        if task != "fer":
            raise ValueError("The flow_mag transition measure is defined for facial data only")
        motion = x.flatten(3).mean(dim=-1)
    else:
        positions = batch.get("landmarks") if task == "fer" else x
        if positions is None:
            raise ValueError("landmark_vel needs node positions in the batch")
        motion = (positions[:, 1:] - positions[:, :-1]).norm(dim=-1)
        if motion.shape[1] != x.shape[1]:
            motion = torch.cat([motion[:, :1], motion], dim=1)
    return motion / motion.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
