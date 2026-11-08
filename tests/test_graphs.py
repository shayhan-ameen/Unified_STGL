import numpy as np
import torch

from models.graphs import (
    CHBMIT_CHANNELS,
    SIENA_CHANNELS,
    build_graph,
    eeg_graph,
    face_graph,
    face_mirror_index,
    normalize_adjacency,
    skeleton_graph,
    skeleton_parents,
)


def _check_binary_symmetric(adj: np.ndarray) -> None:
    assert adj.shape[0] == adj.shape[1]
    assert np.array_equal(adj, adj.T)
    assert np.all(np.diag(adj) == 0)
    assert set(np.unique(adj)).issubset({0.0, 1.0})


def test_face_graph_has_51_nodes_and_74_edges():
    adj = face_graph()
    _check_binary_symmetric(adj)
    assert adj.shape == (51, 51)
    assert int(adj.sum() // 2) == 74


def test_face_mirror_is_an_involution():
    perm = face_mirror_index()
    assert np.array_equal(perm[perm], np.arange(51))
    adj = face_graph()
    assert np.array_equal(adj[np.ix_(perm, perm)], adj)


def test_eeg_graphs():
    for channels, nodes in ((CHBMIT_CHANNELS, 18), (SIENA_CHANNELS, 16)):
        adj = eeg_graph(channels)
        _check_binary_symmetric(adj)
        assert adj.shape == (nodes, nodes)
        assert adj.sum(axis=1).min() >= 1


def test_skeleton_graph_is_a_tree():
    adj = skeleton_graph()
    _check_binary_symmetric(adj)
    assert adj.shape == (25, 25)
    assert int(adj.sum() // 2) == 24
    parents = skeleton_parents()
    assert (parents == np.arange(25)).sum() == 1


def test_normalize_adjacency_adds_self_loops_and_is_symmetric():
    adj = torch.tensor(build_graph("face"))
    norm = normalize_adjacency(adj)
    assert torch.allclose(norm, norm.T)
    assert (norm.diagonal() > 0).all()
    batched = normalize_adjacency(adj.unsqueeze(0).repeat(3, 1, 1))
    assert torch.allclose(batched[1], norm)
