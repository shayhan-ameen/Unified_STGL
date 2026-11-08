"""Training and evaluation entry point.

Examples
--------
    # facial expression recognition on AFEW (official split)
    python main.py --config configs/fer_afew.yaml

    # leave-one-subject-out on CAS(ME)^3 with the 4-class protocol
    python main.py --config configs/fer_casme3_4cls.yaml

    # seizure detection, leave-one-patient-out
    python main.py --config configs/esd_chbmit.yaml

    # skeleton action recognition, joint and bone streams
    python main.py --config configs/har_ntu60_xsub.yaml --set data.stream=joint
    python main.py --config configs/har_ntu60_xsub.yaml --set data.stream=bone

    # ablations through config overrides
    python main.py --config configs/fer_afew.yaml --set model.use_stdegad=false
    python main.py --config configs/fer_afew.yaml --set model.tagcn.modulation=cbam

    # evaluate saved checkpoints only
    python main.py --config configs/fer_afew.yaml --phase test

    # quick end-to-end check on synthetic data (no dataset needed)
    python main.py --config configs/fer_afew.yaml --synthetic --set train.epochs=1

Outputs are written to ``work_dir/<exp_name>/``: ``config.yaml``, ``train.log``,
``history.csv``, one checkpoint per fold, the pooled test predictions and
softmax scores, and ``results.json`` with the final metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import build_model
from utils.batch import SubsetWithTransform, collate, transition_prior
from utils.synthetic import SyntheticDataset
from utils.config import Config, load_config, save_config
from utils.logger import CSVHistory, get_logger
from utils.metrics import compute_metrics, format_metrics, primary_metric
from utils.optim import build_optimizer, build_scheduler
from utils.protocols import make_folds
from utils.seed import seed_everything, worker_init_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a YAML configuration")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    parser.add_argument("--phase", choices=["train", "test"], default="train")
    parser.add_argument("--synthetic", action="store_true", help="use random data of the right shape")
    parser.add_argument("--fold", type=int, default=None, help="run a single fold index only")
    parser.add_argument("--work-dir", default=None, help="override the output directory")
    return parser.parse_args()


def load_benchmark(cfg: Config):
    """Return the benchmark dataset for ``cfg.data`` as a ``GraphSequenceDataset``.

    The benchmark caches are prepared outside this repository; a reader
    module placed under ``data/`` that exposes ``build_dataset(cfg)`` is
    picked up here, otherwise the run must use ``--synthetic``.
    """
    try:
        from data import build_dataset  # provided together with the benchmark caches
    except ImportError as err:
        raise SystemExit(
            "No benchmark reader found under data/. Place the dataset caches and their reader module there, "
            "or run with --synthetic to exercise the pipeline on random data of the right shape."
        ) from err
    return build_dataset(cfg)


def make_loader(dataset, indices, cfg: Config, train: bool) -> DataLoader:
    batch_size = int(cfg.train.batch_size) // int(cfg.train.get("accumulation_steps", 1))
    transform = getattr(dataset, "train_transform", None) if train else None
    return DataLoader(
        SubsetWithTransform(dataset, indices, transform),
        batch_size=max(batch_size, 1),
        shuffle=train,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available(),
    )


def batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, scaler, cfg: Config, device, logger, epoch: int) -> Tuple[float, float]:
    model.train()
    measure, task = cfg.model.tagcn.transition_measure, cfg.data.task
    accumulation = int(cfg.train.get("accumulation_steps", 1))
    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"
    total_loss, correct, seen = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = batch_to_device(batch, device)
        prior = transition_prior(batch, measure, task)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(batch["x"], prior)
            loss = F.cross_entropy(logits, batch["y"])
        scaler.scale(loss / accumulation).backward()
        if step % accumulation == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * batch["y"].numel()
        correct += (logits.argmax(dim=1) == batch["y"]).sum().item()
        seen += batch["y"].numel()
        if step % int(cfg.train.log_interval) == 0 or step == len(loader):
            logger.info(f"epoch {epoch:03d} step {step:04d}/{len(loader):04d} loss {loss.item():.4f}")
    return total_loss / max(seen, 1), 100.0 * correct / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return labels, predictions, softmax scores, and the mean loss."""
    model.eval()
    measure, task = cfg.model.tagcn.transition_measure, cfg.data.task
    labels, preds, scores, total_loss = [], [], [], 0.0
    for batch in loader:
        batch = batch_to_device(batch, device)
        logits, _ = model(batch["x"], transition_prior(batch, measure, task))
        total_loss += F.cross_entropy(logits, batch["y"], reduction="sum").item()
        labels.append(batch["y"].cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        scores.append(logits.softmax(dim=1).cpu().numpy())
    labels, preds, scores = np.concatenate(labels), np.concatenate(preds), np.concatenate(scores)
    return labels, preds, scores, total_loss / max(len(labels), 1)


def run_fold(fold_name: str, train_idx, test_idx, dataset, cfg: Config, device, logger, out_dir: str, phase: str):
    ckpt_path = os.path.join(out_dir, f"{fold_name}.pt")
    model = build_model(cfg).to(device)
    test_loader = make_loader(dataset, test_idx, cfg, train=False)

    if phase == "train":
        train_idx = dataset.training_indices(train_idx, int(cfg.seed)) if hasattr(dataset, "training_indices") else train_idx
        train_loader = make_loader(dataset, train_idx, cfg, train=True)
        optimizer = build_optimizer(cfg.train, model.parameters())
        scheduler = build_scheduler(cfg.train, optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.get("amp", False)) and device.type == "cuda")
        history = CSVHistory(os.path.join(out_dir, f"{fold_name}_history.csv"))
        num_classes = int(cfg.data.num_classes)
        for epoch in range(1, int(cfg.train.epochs) + 1):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, logger, epoch)
            scheduler.step()
            labels, preds, _, test_loss = evaluate(model, test_loader, cfg, device)
            metrics = compute_metrics(labels, preds, num_classes)
            history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss,
                            "train_acc": train_acc, "test_loss": test_loss, **{f"test_{k}": v for k, v in metrics.items()}})
            logger.info(f"[{fold_name}] epoch {epoch:03d}  train loss {train_loss:.4f} acc {train_acc:.2f}  "
                        f"test loss {test_loss:.4f} {format_metrics(metrics)}  ({time.time() - start:.1f}s)")
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "fold": fold_name}, ckpt_path)
    else:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        logger.info(f"[{fold_name}] loaded {ckpt_path}")

    labels, preds, scores, _ = evaluate(model, test_loader, cfg, device)
    return labels, preds, scores, test_idx


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    out_dir = args.work_dir or os.path.join(cfg.work_dir, cfg.exp_name + ("_synthetic" if args.synthetic else ""))
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(log_file=os.path.join(out_dir, f"{args.phase}.log"))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    seed_everything(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"config {args.config}  overrides {args.set}  device {device}")

    if args.synthetic:
        cfg.train.num_workers = 0
        dataset = SyntheticDataset(cfg, num_samples=max(2 * int(cfg.train.batch_size), 8), seed=int(cfg.seed))
        folds = [("synthetic", np.arange(0, len(dataset) * 3 // 4), np.arange(len(dataset) * 3 // 4, len(dataset)))]
    else:
        dataset = load_benchmark(cfg)
        folds = make_folds(cfg.data.protocol, dataset.subjects, dataset.splits, int(cfg.data.get("num_folds", 10)), int(cfg.seed))
    if args.fold is not None:
        folds = [folds[args.fold]]
    logger.info(f"{len(dataset)} samples, {len(folds)} fold(s), protocol {cfg.data.get('protocol', 'synthetic')}")

    all_labels, all_preds, all_scores, all_indices = [], [], [], []
    for fold_name, train_idx, test_idx in folds:
        labels, preds, scores, indices = run_fold(fold_name, train_idx, test_idx, dataset, cfg, device, logger, out_dir, args.phase)
        fold_metrics = compute_metrics(labels, preds, int(cfg.data.num_classes))
        logger.info(f"[{fold_name}] final  {format_metrics(fold_metrics)}")
        all_labels.append(labels); all_preds.append(preds); all_scores.append(scores); all_indices.append(indices)

    labels, preds, scores = np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_scores)
    indices = np.concatenate(all_indices)
    metrics = compute_metrics(labels, preds, int(cfg.data.num_classes))
    headline = primary_metric(cfg.data.task, str(cfg.data.get("protocol", "official")))
    logger.info(f"POOLED over {len(folds)} fold(s): {format_metrics(metrics)}   [headline: {headline} = {metrics[headline]:.2f}]")

    np.savez(os.path.join(out_dir, "predictions.npz"), indices=indices, labels=labels, predictions=preds, scores=scores)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics, "headline": headline, "num_folds": len(folds), "config": cfg.to_dict()}, handle, indent=2)


if __name__ == "__main__":
    main()
