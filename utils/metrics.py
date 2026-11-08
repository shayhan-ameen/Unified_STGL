"""Classification metrics used in the paper.

* ACC / WAR  : N_correct / N_total
* UF1        : mean over classes of F1_c = 2 TP_c / (2 TP_c + FP_c + FN_c)
* UAR        : mean over classes of TP_c / (TP_c + FN_c)
* PRE, REC, SPE, F1 : binary metrics with class 1 (ictal) as the positive class

All metrics are computed from pooled predictions, i.e. predictions of every
fold are concatenated before the confusion matrix is formed.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    """``cm[i, j]`` = number of samples of class ``i`` predicted as class ``j``."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)):
        cm[t, p] += 1
    return cm


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> Dict[str, float]:
    """All metrics (in percent) from pooled predictions."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    total = cm.sum()
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - tp - fp - fn

    with np.errstate(divide="ignore", invalid="ignore"):
        recall_c = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1_c = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)

    metrics = {
        "acc": 100.0 * tp.sum() / max(total, 1),
        "uar": 100.0 * recall_c.mean(),
        "uf1": 100.0 * f1_c.mean(),
    }
    if num_classes == 2:
        pos = 1
        metrics["precision"] = 100.0 * tp[pos] / max(tp[pos] + fp[pos], 1e-12)
        metrics["recall"] = 100.0 * tp[pos] / max(tp[pos] + fn[pos], 1e-12)
        metrics["specificity"] = 100.0 * tn[pos] / max(tn[pos] + fp[pos], 1e-12)
        metrics["f1"] = 100.0 * f1_c[pos]
    return metrics


def primary_metric(task: str, protocol: str) -> str:
    """The metric reported as the headline number for a task / protocol."""
    if task == "esd":
        return "f1"
    if task == "fer" and protocol == "loso":
        return "uar"
    return "acc"


def format_metrics(metrics: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.2f}" for k, v in metrics.items())
