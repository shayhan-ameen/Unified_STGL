import torch

from models.graphs import build_graph, normalize_adjacency
from models.st_degad import GatedDecoder, STDEGaD, pair_features


def _model(use_checkpoint=False):
    return STDEGaD(
        in_channels=16,
        structural_channels=[16, 32],
        contextual_dim=32,
        attention_dim=8,
        hydra_d_state=4,
        hydra_headdim=16,
        rel_dim=8,
        gate_hidden=8,
        pair_hidden=8,
        use_checkpoint=use_checkpoint,
    )


def test_pair_features_window_coverage():
    z = torch.randn(1, 5, 3, 4)
    assert pair_features(z, z, 0).shape == (1, 5, 3, 3, 16)
    assert pair_features(z, z, 1).shape == (1, 4, 3, 3, 16)
    assert pair_features(z, z, -1).shape == (1, 4, 3, 3, 16)


def test_dynamic_adjacency_properties():
    torch.manual_seed(0)
    adj = normalize_adjacency(torch.tensor(build_graph("skeleton")))
    model = _model().eval()
    x = torch.randn(2, 6, 25, 16)
    adj_dynamic, aux = model(x, adj)
    assert adj_dynamic.shape == (2, 25, 25)
    assert torch.allclose(adj_dynamic, adj_dynamic.transpose(1, 2), atol=1e-6)
    assert (adj_dynamic >= 0).all()
    assert (aux["relation"] >= 0).all() and (aux["relation"] <= 1).all()
    assert torch.allclose(aux["pair_weights"].sum(dim=1), torch.ones(2, 25, 25), atol=1e-5)
    # Samples get different graphs.
    assert not torch.allclose(adj_dynamic[0], adj_dynamic[1])


def test_checkpointed_decoder_matches_dense_path():
    torch.manual_seed(0)
    dense = GatedDecoder(32, radius=1, rel_dim=8, gate_hidden=8, pair_hidden=8, use_checkpoint=False)
    ckpt = GatedDecoder(32, radius=1, rel_dim=8, gate_hidden=8, pair_hidden=8, use_checkpoint=True)
    ckpt.load_state_dict(dense.state_dict())
    z_s, z_c = torch.randn(2, 5, 7, 32, requires_grad=True), torch.randn(2, 5, 7, 32, requires_grad=True)
    a1, _ = dense(z_s, z_c)
    a2, _ = ckpt(z_s, z_c)
    assert torch.allclose(a1, a2, atol=1e-6)
    g1 = torch.autograd.grad(a1.sum(), z_s, retain_graph=True)[0]
    g2 = torch.autograd.grad(a2.sum(), z_s)[0]
    assert torch.allclose(g1, g2, atol=1e-6)
