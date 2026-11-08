"""EEG seizure-detection dataset over the segment cache.

The cache holds one ``.npz`` per EDF recording with

    x : ``[S, T, N, n_w]`` windowed, z-scored bipolar signals of S segments
    y : ``[S]`` segment labels (1 = ictal, 0 = interictal)

and an ``index.csv`` with the columns ``file, path, patient, num_segments``.
Each segment is one sample; the patient id is the subject used by the
leave-one-patient-out protocol.  ``training_indices`` balances ictal and
interictal segments 1:1 within every training patient, leaving the test
patient untouched.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

import numpy as np
import torch

from .base import GraphSequenceDataset


class ESDDataset(GraphSequenceDataset):
    """Epileptic seizure segments as graph sequences."""

    def __init__(self, cfg) -> None:
        self.root = cfg.data.root
        self.num_windows = int(cfg.data.num_windows)
        self.window_samples = int(cfg.data.window_samples)
        self.balance = bool(cfg.data.get("balance_train", True))
        index_path = os.path.join(self.root, "index.csv")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"No index.csv under {self.root}; see the data preparation section of the README")
        self._arrays: Dict[str, np.lib.npyio.NpzFile] = {}
        records: List[Dict[str, Any]] = []
        with open(index_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = os.path.join(self.root, row["path"])
                labels = np.load(path)["y"]
                for k, y in enumerate(labels):
                    records.append({"path": path, "segment": k, "subject": row["patient"], "split": "", "y": int(y), "id": f"{row['file']}_{k}"})
        if not records:
            raise RuntimeError(f"No segments found under {self.root}")
        super().__init__(records)

    def _file(self, path: str):
        if path not in self._arrays:
            self._arrays[path] = np.load(path, mmap_mode="r")
        return self._arrays[path]

    def load(self, record: Dict[str, Any]) -> Dict[str, Any]:
        x = np.array(self._file(record["path"])["x"][record["segment"]], dtype=np.float32)
        if x.shape[0] != self.num_windows or x.shape[-1] != self.window_samples:
            raise ValueError(f"{record['path']} holds segments of shape {x.shape}, expected "
                             f"[{self.num_windows}, N, {self.window_samples}]")
        return {"x": torch.from_numpy(x), "y": torch.tensor(record["y"], dtype=torch.long),
                "subject": record["subject"], "id": record["id"]}

    def training_indices(self, indices: np.ndarray, seed: int = 0) -> np.ndarray:
        """Down-sample the majority class per patient so that ictal : interictal = 1 : 1."""
        if not self.balance:
            return indices
        rng = np.random.default_rng(seed)
        subjects = np.asarray(self.subjects)[indices]
        labels = np.asarray(self.labels)[indices]
        kept = []
        for patient in np.unique(subjects):
            member = indices[subjects == patient]
            member_labels = labels[subjects == patient]
            ictal, inter = member[member_labels == 1], member[member_labels == 0]
            count = min(len(ictal), len(inter))
            if count == 0:
                continue
            kept.append(rng.choice(ictal, count, replace=False))
            kept.append(rng.choice(inter, count, replace=False))
        return np.sort(np.concatenate(kept)) if kept else indices
