import numpy as np
import pytest

from utils.metrics import compute_metrics, confusion_matrix
from utils.protocols import fuse_scores, make_folds


def test_metrics_against_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=200)
    y_pred = np.where(rng.random(200) < 0.7, y_true, rng.integers(0, 4, size=200))
    m = compute_metrics(y_true, y_pred, 4)
    assert np.isclose(m["acc"], 100 * sklearn.accuracy_score(y_true, y_pred))
    assert np.isclose(m["uar"], 100 * sklearn.recall_score(y_true, y_pred, average="macro"))
    assert np.isclose(m["uf1"], 100 * sklearn.f1_score(y_true, y_pred, average="macro"))


def test_binary_metrics():
    y_true = [0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 1, 0, 1]
    m = compute_metrics(y_true, y_pred, 2)
    assert np.isclose(m["precision"], 100 * 3 / 4)
    assert np.isclose(m["recall"], 100 * 3 / 4)
    assert np.isclose(m["specificity"], 100 * 2 / 3)
    assert confusion_matrix(y_true, y_pred, 2).tolist() == [[2, 1], [1, 3]]


def test_folds_are_subject_disjoint():
    subjects = [f"s{i % 7}" for i in range(50)]
    splits = ["train" if i < 30 else "test" for i in range(50)]
    for protocol, expected in (("kfold", 3), ("loso", 7), ("official", 1)):
        folds = make_folds(protocol, subjects, splits, num_folds=3)
        assert len(folds) == expected
        for _, train, test in folds:
            assert len(set(train) & set(test)) == 0
            if protocol != "official":
                assert not set(np.asarray(subjects)[train]) & set(np.asarray(subjects)[test])


def test_fuse_scores_averages():
    a = np.array([[0.9, 0.1], [0.2, 0.8]])
    b = np.array([[0.5, 0.5], [0.6, 0.4]])
    assert np.allclose(fuse_scores([a, b]), (a + b) / 2)
