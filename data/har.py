"""NTU RGB+D skeleton dataset over the stream cache.

The cache holds, per benchmark protocol,

    <root>/<protocol>/<split>_joint.npy   ``[S, T, 25, 3]`` spine-centred joint coordinates
    <root>/<protocol>/<split>_bone.npy    ``[S, T, 25, 3]`` bone displacements
    <root>/<protocol>/<split>_labels.npy  ``[S]`` action classes
    <root>/<protocol>/<split>_names.txt   sample names (one per line)

for ``split`` in ``train`` / ``test``.  ``HARDataset`` exposes the stream
selected by ``cfg.data.stream`` and marks each sample with its split so the
``official`` protocol can be applied.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import torch

from .base import GraphSequenceDataset


class HARDataset(GraphSequenceDataset):
    """Skeleton action sequences (one stream) with the NTU partition."""

    def __init__(self, cfg) -> None:
        self.root = os.path.join(cfg.data.root, cfg.data.protocol)
        self.stream = str(cfg.data.get("stream", "joint")).lower()
        self.num_frames = int(cfg.data.num_frames)
        if self.stream not in ("joint", "bone"):
            raise ValueError("data.stream must be 'joint' or 'bone'")
        self._data: Dict[str, np.ndarray] = {}
        records: List[Dict[str, Any]] = []
        for split in ("train", "test"):
            path = os.path.join(self.root, f"{split}_{self.stream}.npy")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} not found; see the data preparation section of the README")
            self._data[split] = np.load(path, mmap_mode="r")
            labels = np.load(os.path.join(self.root, f"{split}_labels.npy"))
            with open(os.path.join(self.root, f"{split}_names.txt"), encoding="utf-8") as handle:
                names = [line.strip() for line in handle if line.strip()]
            for k, (name, y) in enumerate(zip(names, labels)):
                records.append({"split": split, "index": k, "subject": name[8:12], "y": int(y), "id": name})
        super().__init__(records)

    def load(self, record: Dict[str, Any]) -> Dict[str, Any]:
        x = np.array(self._data[record["split"]][record["index"]], dtype=np.float32)
        if x.shape[0] != self.num_frames:
            raise ValueError(f"cached sequences have {x.shape[0]} frames, config asks for {self.num_frames}")
        return {"x": torch.from_numpy(x), "y": torch.tensor(record["y"], dtype=torch.long),
                "subject": record["subject"], "id": record["id"]}
