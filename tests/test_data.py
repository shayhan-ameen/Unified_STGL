"""Round-trip checks of the three cache formats through the dataset readers."""

import csv
import os

import numpy as np
import pytest
import torch

from data import build_dataset
from data.base import collate, transition_prior
from data.fer import label_to_index
from utils.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(name, overrides):
    return load_config(os.path.join(ROOT, "configs", name), overrides)


def test_fer_label_maps():
    assert label_to_index("afew", "Happy") == 3
    assert label_to_index("casme3", "disgust", "4cls") == 1
    assert label_to_index("casme3", "disgust", "7cls") == 3
    assert label_to_index("smic_hs", "surprise") == 2
    assert label_to_index("oulu_vl", "Neutral") == -1


def test_fer_cache_roundtrip_and_augmentation(tmp_path):
    rng = np.random.default_rng(0)
    os.makedirs(tmp_path / "clips")
    with open(tmp_path / "index.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "path", "subject", "label", "split"])
        writer.writeheader()
        for k in range(6):
            np.savez(tmp_path / "clips" / f"c{k}.npz", patches=rng.gamma(2.0, 0.5, (8, 51, 7, 7)).astype(np.float32),
                     landmarks=rng.uniform(40, 180, (9, 51, 2)).astype(np.float32), keyframes=np.arange(9))
            writer.writerow({"id": f"c{k}", "path": f"clips/c{k}.npz", "subject": f"s{k % 3}", "label": ["Angry", "Happy", "Sad"][k % 3],
                             "split": "train" if k < 4 else "val"})
    cfg = _cfg("fer_afew.yaml", [f"data.root={tmp_path.as_posix()}"])
    dataset = build_dataset(cfg)
    assert len(dataset) == 6 and dataset.subjects[0] == "s0" and dataset.splits[-1] == "val"
    sample = dataset[0]
    assert sample["x"].shape == (8, 51, 7, 7) and sample["landmarks"].shape == (9, 51, 2)
    augmented = dataset.train_transform(sample)
    assert augmented["x"].shape == sample["x"].shape and (augmented["x"] >= 0).all()
    batch = collate([dataset[0], dataset[1]])
    for measure in ("flow_mag", "landmark_vel"):
        prior = transition_prior(batch, measure, "fer")
        assert prior.shape == (2, 8, 51) and prior.min() >= 0 and prior.max() <= 1


def test_esd_cache_roundtrip_and_balancing(tmp_path):
    rng = np.random.default_rng(0)
    os.makedirs(tmp_path / "recordings")
    rows = []
    for patient, (ictal, inter) in {"chb01": (3, 9), "chb02": (2, 5)}.items():
        x = rng.normal(size=(ictal + inter, 7, 18, 256)).astype(np.float32)
        y = np.array([1] * ictal + [0] * inter)
        np.savez(tmp_path / "recordings" / f"{patient}_01.npz", x=x, y=y)
        rows.append({"file": f"{patient}_01", "path": f"recordings/{patient}_01.npz", "patient": patient, "num_segments": len(y)})
    with open(tmp_path / "index.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "path", "patient", "num_segments"])
        writer.writeheader()
        writer.writerows(rows)
    cfg = _cfg("esd_chbmit.yaml", [f"data.root={tmp_path.as_posix()}"])
    dataset = build_dataset(cfg)
    assert len(dataset) == 19 and dataset[0]["x"].shape == (7, 18, 256)
    balanced = dataset.training_indices(np.arange(len(dataset)))
    labels = np.asarray(dataset.labels)[balanced]
    assert len(balanced) == 10 and (labels == 1).sum() == (labels == 0).sum()


def test_har_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    folder = tmp_path / "xsub"
    os.makedirs(folder)
    for split, count in (("train", 5), ("test", 3)):
        np.save(folder / f"{split}_joint.npy", rng.normal(size=(count, 64, 25, 3)).astype(np.float32))
        np.save(folder / f"{split}_bone.npy", rng.normal(size=(count, 64, 25, 3)).astype(np.float32))
        np.save(folder / f"{split}_labels.npy", rng.integers(0, 60, count))
        with open(folder / f"{split}_names.txt", "w") as handle:
            handle.write("\n".join(f"S001C001P{k:03d}R001A{k + 1:03d}" for k in range(count)) + "\n")
    cfg = _cfg("har_ntu60_xsub.yaml", [f"data.root={tmp_path.as_posix()}", "data.stream=bone"])
    dataset = build_dataset(cfg)
    assert len(dataset) == 8 and dataset[0]["x"].shape == (64, 25, 3)
    assert dataset.splits.count("train") == 5
