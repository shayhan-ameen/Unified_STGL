import glob
import os

import pytest
import torch
import torch.nn.functional as F

from models import build_model
from utils.config import load_config
from utils.synthetic import make_batch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(p for p in glob.glob(os.path.join(ROOT, "configs", "*.yaml")) if not p.endswith("base.yaml"))

SMALL = [
    "model.structural.channels=[16,32]",
    "model.contextual.dim=32",
    "model.contextual.attention_dim=8",
    "model.contextual.hydra.headdim=16",
    "model.contextual.hydra.d_state=4",
    "model.tagcn.channels=[16,32]",
    "model.decoder.rel_dim=8",
    "model.decoder.gate_hidden=8",
    "model.decoder.pair_hidden=8",
]


def _run(cfg, batch_size=2):
    model = build_model(cfg).train()
    x, y = make_batch(cfg, batch_size)
    logits, aux = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    assert logits.shape == (batch_size, cfg.data.num_classes)
    assert torch.isfinite(loss)
    return model, aux


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.basename(p) for p in CONFIGS])
def test_every_config_builds_and_trains_one_step(path):
    overrides = SMALL + (["data.num_frames=8"] if "har" in path else [])
    _run(load_config(path, overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        ["model.use_stdegad=false"],
        ["model.use_sttagcn=false"],
        ["model.use_stdegad=false", "model.use_sttagcn=false"],
        ["model.structural.attention=channel"],
        ["model.tagcn.modulation=cbam"],
        ["model.tagcn.transition_measure=l2"],
    ],
)
def test_ablation_switches(overrides):
    cfg = load_config(os.path.join(ROOT, "configs", "fer_afew.yaml"), SMALL + overrides)
    _run(cfg)


def test_aux_outputs_have_expected_shapes():
    cfg = load_config(os.path.join(ROOT, "configs", "fer_afew.yaml"), SMALL)
    model, aux = _run(cfg)
    assert aux["adjacency_dynamic"].shape == (2, 51, 51)
    assert len(aux["alpha_spatial"]) == 2 and aux["alpha_spatial"][-1].shape == (2, 8, 51)
    assert len(aux["alpha_temporal"]) == 2 and aux["alpha_temporal"][-1].shape == (2, 8)
    assert all(0 <= a.min() and a.max() <= 1 for a in aux["alpha_spatial"] + aux["alpha_temporal"])
