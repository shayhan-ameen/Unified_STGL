"""Dataset package: synthetic samples plus the facial, EEG, and skeleton readers.

``build_dataset(cfg)`` returns the dataset for ``cfg.data.task`` with every
sample of the benchmark; the evaluation protocol in ``utils.protocols`` then
decides which samples form the training and test folds.
"""

from __future__ import annotations

from .base import GraphSequenceDataset, collate, transition_prior


def build_dataset(cfg) -> GraphSequenceDataset:
    task = cfg.data.task.lower()
    if task == "fer":
        from .fer import FERDataset
        return FERDataset(cfg)
    if task == "esd":
        from .esd import ESDDataset
        return ESDDataset(cfg)
    if task == "har":
        from .har import HARDataset
        return HARDataset(cfg)
    raise ValueError(f"Unknown task '{cfg.data.task}'")


__all__ = ["GraphSequenceDataset", "build_dataset", "collate", "transition_prior"]
