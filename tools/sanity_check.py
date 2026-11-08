"""Build every configuration and every ablation variant, then run one forward
and backward pass on a small synthetic batch.

Usage
-----
    python tools/sanity_check.py                # all configs in configs/, CPU
    python tools/sanity_check.py --device cuda  # on the GPU
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.synthetic import make_batch  # noqa: E402
from models import build_model  # noqa: E402
from utils.config import load_config  # noqa: E402

ABLATIONS = [
    ("full model", []),
    ("w/o ST-DEGaD", ["model.use_stdegad=false"]),
    ("w/o ST-TAGCN", ["model.use_sttagcn=false"]),
    ("w/o both (ST-GCN)", ["model.use_stdegad=false", "model.use_sttagcn=false"]),
    ("L = 1", ["model.structural.channels=[128]"]),
    ("L = 3", ["model.structural.channels=[64,64,128]"]),
    ("K = 1", ["model.tagcn.channels=[64]"]),
    ("K = 5", ["model.tagcn.channels=[64,128,128,256,256]"]),
    ("attention = none", ["model.structural.attention=none"]),
    ("attention = channel", ["model.structural.attention=channel"]),
    ("attention = frame", ["model.structural.attention=frame"]),
    ("transition = l1", ["model.tagcn.transition_measure=l1"]),
    ("transition = l2", ["model.tagcn.transition_measure=l2"]),
    ("transition = landmark_vel", ["model.tagcn.transition_measure=landmark_vel"]),
    ("modulation = se", ["model.tagcn.modulation=se"]),
    ("modulation = cbam", ["model.tagcn.modulation=cbam"]),
    ("modulation = st_selfattn", ["model.tagcn.modulation=st_selfattn"]),
    ("modulation = static_learnable", ["model.tagcn.modulation=static_learnable"]),
    ("decoder checkpointing", ["model.decoder.checkpoint=true"]),
]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_once(cfg, device: torch.device, batch_size: int = 2) -> tuple[int, float]:
    """Forward + backward on a synthetic batch; returns (#params, seconds)."""
    model = build_model(cfg).to(device).train()
    x, y = make_batch(cfg, batch_size)
    x, y = x.to(device), y.to(device)
    prior = None
    if cfg.model.tagcn.transition_measure in ("flow_mag", "landmark_vel"):
        # Input-level transition prior: mean patch magnitude or node velocity, scaled to [0, 1].
        motion = x.flatten(3).mean(dim=-1) if cfg.data.task == "fer" else (x[:, 1:] - x[:, :-1]).norm(dim=-1)
        if motion.shape[1] != x.shape[1]:
            motion = torch.cat([motion[:, :1], motion], dim=1)
        prior = motion / motion.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    start = time.time()
    logits, aux = model(x, prior)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    assert logits.shape == (batch_size, cfg.data.num_classes)
    assert torch.isfinite(loss)
    assert aux["adjacency_dynamic"].shape == (batch_size, cfg.data.num_nodes, cfg.data.num_nodes)
    return count_parameters(model), time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--configs", default=os.path.join(ROOT, "configs"))
    parser.add_argument("--ablation-config", default="fer_afew.yaml", help="config used for the ablation variants")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    print(f"{'config':<32}{'params':>12}{'fwd+bwd (s)':>14}")
    for path in sorted(glob.glob(os.path.join(args.configs, "*.yaml"))):
        if os.path.basename(path) == "base.yaml":
            continue
        cfg = load_config(path)
        params, seconds = run_once(cfg, device)
        print(f"{os.path.basename(path):<32}{params:>12,}{seconds:>14.2f}")

    print()
    print(f"{'ablation variant':<32}{'params':>12}{'fwd+bwd (s)':>14}")
    base = os.path.join(args.configs, args.ablation_config)
    for name, overrides in ABLATIONS:
        cfg = load_config(base, overrides)
        params, seconds = run_once(cfg, device)
        print(f"{name:<32}{params:>12,}{seconds:>14.2f}")
    print("\nAll configurations ran forward and backward successfully.")


if __name__ == "__main__":
    main()
