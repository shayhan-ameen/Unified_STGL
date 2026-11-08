"""Evaluation protocols: how a dataset is partitioned into training and test folds.

* ``official``  the dataset's own train / test partition (AFEW, NTU protocols)
* ``kfold``     subject-independent K-fold cross-validation (Oulu-VL)
* ``loso``      leave-one-subject-out (CAS(ME)^3, SMIC-HS)
* ``lopo``      leave-one-patient-out, identical to LOSO with patients as subjects
* ``xsub`` / ``xview`` / ``xset``  NTU RGB+D partitions, resolved by the reader
                into ``split`` fields and treated as ``official``

Every protocol yields ``(fold_name, train_indices, test_indices)`` triples.
Two-stream fusion for skeleton data averages the saved softmax scores.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import numpy as np

Fold = Tuple[str, np.ndarray, np.ndarray]


def official_split(splits: Sequence[str]) -> Iterator[Fold]:
    splits = np.asarray(splits)
    train = np.where(splits == "train")[0]
    test = np.where(np.isin(splits, ["test", "val"]))[0]
    if len(train) == 0 or len(test) == 0:
        raise ValueError("The official protocol needs samples marked 'train' and 'test' (or 'val')")
    yield "official", train, test


def subject_kfold(subjects: Sequence[str], num_folds: int, seed: int = 0) -> Iterator[Fold]:
    """Assign whole subjects to folds so that identities never cross the split."""
    subjects = np.asarray(subjects)
    unique = np.array(sorted(set(subjects.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    for k, held_out in enumerate(np.array_split(unique, num_folds)):
        test = np.where(np.isin(subjects, held_out))[0]
        train = np.where(~np.isin(subjects, held_out))[0]
        yield f"fold{k:02d}", train, test


def leave_one_subject_out(subjects: Sequence[str]) -> Iterator[Fold]:
    subjects = np.asarray(subjects)
    for held_out in sorted(set(subjects.tolist())):
        test = np.where(subjects == held_out)[0]
        train = np.where(subjects != held_out)[0]
        yield f"subject_{held_out}", train, test


def make_folds(protocol: str, subjects: Sequence[str], splits: Sequence[str], num_folds: int = 10, seed: int = 0) -> List[Fold]:
    protocol = protocol.lower()
    if protocol in ("official", "xsub", "xview", "xset"):
        return list(official_split(splits))
    if protocol == "kfold":
        return list(subject_kfold(subjects, num_folds, seed))
    if protocol in ("loso", "lopo"):
        return list(leave_one_subject_out(subjects))
    raise ValueError(f"Unknown protocol '{protocol}'")


def fuse_scores(score_arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Average softmax score matrices ``[num_samples, num_classes]`` from several streams."""
    stacked = np.stack([np.asarray(s, dtype=np.float64) for s in score_arrays])
    if any(s.shape != stacked[0].shape for s in stacked):
        raise ValueError("All score arrays must have the same shape")
    return stacked.mean(axis=0)
