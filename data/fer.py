"""Facial expression datasets over the landmark-motion cache.

The cache holds, for every clip, one ``.npz`` with

    patches    : ``[T, 51, Sz, Sz]`` optical-flow magnitude patches
    landmarks  : ``[T + 1, 51, 2]`` aligned landmark coordinates of the keyframes
    keyframes  : indices of the selected frames in the original clip

and an ``index.csv`` with the columns ``id, path, subject, label, split``.
``FERDataset`` reads the index, maps the textual labels to class indices for
the configured label set, and loads the arrays on demand.

Label sets
----------
afew      7 : anger, disgust, fear, happiness, sadness, surprise, neutral
oulu_vl   6 : anger, disgust, fear, happiness, sadness, surprise
casme3    7 : happiness, sadness, anger, disgust, fear, surprise, others
          4 : positive, negative (anger, disgust, fear, sadness), surprise, others
smic_hs   3 : positive, negative, surprise
ck_plus   6 : as oulu_vl (cross-dataset macro-expression protocol)
casme2    3 : positive (happiness), negative (disgust, repression, sadness, fear), surprise
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

import numpy as np
import torch

from .base import GraphSequenceDataset
from .transforms import FaceAugmentation

# Canonical spelling of expression names found in the different annotation files.
_SYNONYMS = {
    "angry": "anger", "anger": "anger",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "fearful": "fear", "afraid": "fear",
    "happy": "happiness", "happiness": "happiness", "joy": "happiness",
    "sad": "sadness", "sadness": "sadness",
    "surprise": "surprise", "surprised": "surprise",
    "neutral": "neutral", "contempt": "contempt", "others": "others", "other": "others",
    "repression": "repression", "tense": "tense",
    "positive": "positive", "negative": "negative",
}

LABEL_SETS: Dict[str, List[str]] = {
    "afew": ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"],
    "oulu_vl": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
    "ck_plus": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
    "casme3_7cls": ["happiness", "sadness", "anger", "disgust", "fear", "surprise", "others"],
    "casme3_4cls": ["positive", "negative", "surprise", "others"],
    "smic_hs": ["positive", "negative", "surprise"],
    "casme2": ["positive", "negative", "surprise"],
}

# Emotion -> coarse class for the 3- and 4-class micro-expression protocols.
_COARSE = {
    "happiness": "positive", "positive": "positive",
    "anger": "negative", "disgust": "negative", "fear": "negative", "sadness": "negative",
    "repression": "negative", "negative": "negative",
    "surprise": "surprise", "others": "others",
}


def canonical_label(label: str) -> str:
    key = str(label).strip().lower()
    return _SYNONYMS.get(key, key)


def label_set_name(dataset: str, label_set: str = "") -> str:
    dataset = dataset.lower()
    if dataset == "casme3":
        return f"casme3_{label_set or '4cls'}"
    return dataset


def label_to_index(dataset: str, label: str, label_set: str = "") -> int:
    """Map an annotation label to the class index of the configured label set, or -1."""
    name = label_set_name(dataset, label_set)
    classes = LABEL_SETS[name]
    label = canonical_label(label)
    if name in ("casme3_4cls", "smic_hs", "casme2"):
        label = _COARSE.get(label, label)
    return classes.index(label) if label in classes else -1


def read_index(root: str) -> List[Dict[str, str]]:
    path = os.path.join(root, "index.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No index.csv under {root}; see the data preparation section of the README")
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class FERDataset(GraphSequenceDataset):
    """Facial expression clips as graph sequences."""

    def __init__(self, cfg) -> None:
        self.root = cfg.data.root
        self.dataset = cfg.data.dataset
        self.label_set = str(cfg.data.get("label_set", ""))
        self.num_keyframes = int(cfg.data.num_keyframes)
        records = []
        for row in read_index(self.root):
            y = label_to_index(self.dataset, row["label"], self.label_set)
            if y < 0:
                continue  # label outside the configured class set
            records.append({
                "id": row["id"],
                "path": os.path.join(self.root, row["path"]) if not os.path.isabs(row["path"]) else row["path"],
                "subject": row.get("subject", row["id"]),
                "split": row.get("split", ""),
                "y": y,
            })
        if not records:
            raise RuntimeError(f"No samples with labels in {label_set_name(self.dataset, self.label_set)} under {self.root}")
        super().__init__(records)
        if bool(cfg.data.get("augment", False)):
            self.train_transform = FaceAugmentation(seed=int(cfg.seed))

    def load(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with np.load(record["path"]) as data:
            patches = data["patches"].astype(np.float32)
            landmarks = data["landmarks"].astype(np.float32)
        if patches.shape[0] != self.num_keyframes:
            raise ValueError(
                f"{record['path']} holds {patches.shape[0]} flow intervals but the config asks for "
                f"{self.num_keyframes}; the cache must be built with T = {self.num_keyframes}"
            )
        return {
            "x": torch.from_numpy(patches),
            "landmarks": torch.from_numpy(landmarks),
            "y": torch.tensor(record["y"], dtype=torch.long),
            "subject": record["subject"],
            "id": record["id"],
        }
