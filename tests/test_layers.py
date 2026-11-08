import torch

from models.graphs import build_graph, normalize_adjacency
from models.layers import SGLayer, SpatialGraphConv, TemporalConv
from models.stda import build_attention


def test_sglayer_shapes():
    layer = SGLayer(64, 128, kernel_size=3)
    out = layer(torch.randn(2, 64, 8, 51))
    assert out.shape == (2, 128, 8, 51)


def test_spatial_graph_conv_shared_and_batched_adjacency():
    adj = normalize_adjacency(torch.tensor(build_graph("face")))
    conv = SpatialGraphConv(64, 128)
    x = torch.randn(2, 8, 51, 64)
    shared = conv(x, adj)
    batched = conv(x, adj.unsqueeze(0).repeat(2, 1, 1))
    assert shared.shape == (2, 8, 51, 128)
    assert torch.allclose(shared, batched, atol=1e-5)


def test_temporal_conv_preserves_time_and_nodes():
    out = TemporalConv(128, 128, kernel_size=3)(torch.randn(2, 8, 51, 128))
    assert out.shape == (2, 8, 51, 128)


def test_attention_variants():
    x = torch.randn(2, 8, 51, 128)
    for kind in ("stda", "channel", "frame", "none"):
        out, att = build_attention(kind, 128)(x)
        assert out.shape == x.shape
        assert att.shape == x.shape
        assert att.min() >= 0 and att.max() <= 1
