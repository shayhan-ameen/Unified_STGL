"""Parameter counts, FLOPs, latency, throughput, and peak memory per module.

Usage
-----
    python tools/complexity.py --config configs/fer_afew.yaml [--device cuda] [--repeats 100]

FLOPs are measured with fvcore at batch size 1 and reported as GFLOPs
(one fused multiply-add counted as one FLOP, the convention used in the paper).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import build_model  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.synthetic import make_batch  # noqa: E402


def count_parameters(module: torch.nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


def module_table(model) -> list[tuple[str, torch.nn.Module]]:
    rows = [("Input encoder (graph representation)", model.encoder)]
    if model.st_degad is not None:
        rows += [
            ("ST-DEGaD", model.st_degad),
            ("  Structural Encoder", model.st_degad.structural),
            ("  Contextual Encoder", model.st_degad.contextual),
            ("  Gated Decoder", model.st_degad.decoder),
        ]
    rows += [("ST-TAGCN", model.st_tagcn), ("Classifier", model.classifier)]
    return rows


def measure_flops(model, x: torch.Tensor) -> dict:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return {}
    analysis = FlopCountAnalysis(model, (x,))
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    by_module = analysis.by_module()
    return {name: by_module.get(name, 0.0) / 1e9 for name in ["", "encoder", "st_degad", "st_degad.structural",
                                                                "st_degad.contextual", "st_degad.decoder",
                                                                "st_tagcn", "classifier"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    device = torch.device(args.device)
    model = build_model(cfg).to(device).eval()
    x, _ = make_batch(cfg, batch_size=1)
    x = x.to(device)

    flops = measure_flops(model, x)
    name_to_key = {
        "Input encoder (graph representation)": "encoder", "ST-DEGaD": "st_degad",
        "  Structural Encoder": "st_degad.structural", "  Contextual Encoder": "st_degad.contextual",
        "  Gated Decoder": "st_degad.decoder", "ST-TAGCN": "st_tagcn", "Classifier": "classifier",
    }
    print(f"Configuration: {args.config}   input shape: {tuple(x.shape)}")
    print(f"{'Module':<40}{'Params (M)':>12}{'GFLOPs':>12}")
    for name, module in module_table(model):
        gflops = flops.get(name_to_key[name], float('nan')) if flops else float("nan")
        print(f"{name:<40}{count_parameters(module):>12.4f}{gflops:>12.4f}")
    total_flops = flops.get("", float("nan")) if flops else float("nan")
    print(f"{'Total':<40}{count_parameters(model):>12.4f}{total_flops:>12.4f}")

    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    mean_ms = 1000 * sum(times) / len(times)
    std_ms = 1000 * (sum((t * 1000 - mean_ms) ** 2 for t in times) / len(times)) ** 0.5 / 1000
    print(f"\nLatency: {mean_ms:.2f} +/- {std_ms:.2f} ms per sample   throughput: {1000 / mean_ms:.2f} samples/s")
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
