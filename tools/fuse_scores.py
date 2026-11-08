"""Two-stream fusion for skeleton-based action recognition.

Averages the softmax scores saved by ``main.py`` for the joint and bone
streams and reports the fused accuracy.

Usage
-----
    python tools/fuse_scores.py work_dir/har_ntu60_xsub_joint/predictions.npz \
                                work_dir/har_ntu60_xsub_bone/predictions.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.metrics import compute_metrics, format_metrics  # noqa: E402
from utils.protocols import fuse_scores  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", nargs="+", help="predictions.npz files, one per stream")
    args = parser.parse_args()

    runs = [np.load(p) for p in args.predictions]
    reference = runs[0]["indices"]
    scores = []
    for run in runs:
        order = np.argsort(run["indices"])
        if not np.array_equal(run["indices"][order], np.sort(reference)):
            raise ValueError("The streams were evaluated on different samples")
        scores.append(run["scores"][order])
    labels = runs[0]["labels"][np.argsort(reference)]
    fused = fuse_scores(scores)
    metrics = compute_metrics(labels, fused.argmax(axis=1), fused.shape[1])
    for path, run in zip(args.predictions, runs):
        single = compute_metrics(run["labels"], run["predictions"], fused.shape[1])
        print(f"{os.path.basename(os.path.dirname(path)):<32} {format_metrics(single)}")
    print(f"{'fused':<32} {format_metrics(metrics)}")


if __name__ == "__main__":
    main()
