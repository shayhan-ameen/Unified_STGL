"""Predefined graph topologies for the three human-centric modalities.

Each builder returns a binary, symmetric adjacency matrix with a zero diagonal,
A in {0,1}^{N x N}.  Self-connections are introduced later by the graph
convolution of the Structural Encoder and by the adjacency refinement of the
Gated Decoder, exactly as in the paper.

* ``face_graph``      51 inner-face landmarks (68-point layout without the 17
                      contour points) with intra-component chains/cycles and
                      cross-component connections.
* ``eeg_graph``       Bipolar EEG channels placed at the midpoint of their two
                      electrodes in the 10-20 layout; two channels are connected
                      when their normalised Euclidean distance is at most 0.5.
* ``skeleton_graph``  25 Kinect-v2 body joints connected by physical bones.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def edges_to_adjacency(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Build an undirected binary adjacency matrix from an edge list."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        if i == j:
            continue
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


def normalize_adjacency(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """Symmetric degree normalisation D^{-1/2} (A + I) D^{-1/2}.

    Works for a single ``[N, N]`` matrix or a batch ``[B, N, N]``.
    """
    if add_self_loops:
        eye = torch.eye(adj.shape[-1], dtype=adj.dtype, device=adj.device)
        adj = adj + eye
    degree = adj.sum(dim=-1)
    inv_sqrt = degree.clamp(min=1e-12).rsqrt()
    return inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)


# --------------------------------------------------------------------------- #
# Facial landmark graph
# --------------------------------------------------------------------------- #

# Landmark groups in the 68-point dlib convention.  The jaw line (0-16) is
# removed, so every retained index is shifted by -17.
_FACE_OFFSET = 17
_FACE_GROUPS_68: Dict[str, List[int]] = {
    "right_brow": list(range(17, 22)),
    "left_brow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_bottom": list(range(31, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_lip": list(range(48, 60)),
    "inner_lip": list(range(60, 68)),
}


def _chain(indices: List[int], closed: bool) -> List[Tuple[int, int]]:
    edges = [(indices[k], indices[k + 1]) for k in range(len(indices) - 1)]
    if closed:
        edges.append((indices[-1], indices[0]))
    return edges


def face_edges_68() -> List[Tuple[int, int]]:
    """Edge list of the facial graph in 68-point indexing."""
    g = _FACE_GROUPS_68
    edges: List[Tuple[int, int]] = []
    # Intra-component connections.
    edges += _chain(g["right_brow"], closed=False)
    edges += _chain(g["left_brow"], closed=False)
    edges += _chain(g["nose_bridge"], closed=False)
    edges += [(30, 33)]  # nose tip to the centre of the nose bottom
    edges += _chain(g["nose_bottom"], closed=False)
    edges += _chain(g["right_eye"], closed=True)
    edges += _chain(g["left_eye"], closed=True)
    edges += _chain(g["outer_lip"], closed=True)
    edges += _chain(g["inner_lip"], closed=True)
    # Cross-component connections (nose bridge <-> brows / eyes, brows <-> eyes,
    # nose bottom <-> eyes / mouth, outer lip <-> inner lip).
    edges += [
        (27, 38), (27, 39), (27, 40), (27, 21), (27, 22), (27, 42), (27, 43), (27, 47),
        (17, 36), (21, 39), (22, 42), (26, 45), (48, 60), (54, 64),
        (36, 31), (41, 31), (40, 31), (39, 31), (48, 31), (49, 31),
        (42, 35), (47, 35), (46, 35), (45, 35), (54, 35), (53, 35),
    ]
    return edges


def face_graph() -> np.ndarray:
    """Binary adjacency of the 51-node facial graph (74 undirected edges)."""
    edges = [(i - _FACE_OFFSET, j - _FACE_OFFSET) for i, j in face_edges_68()]
    return edges_to_adjacency(51, edges)


def face_mirror_index() -> np.ndarray:
    """Permutation mapping each of the 51 landmarks to its horizontal mirror.

    Used by the horizontal-flip augmentation so that a mirrored face keeps the
    semantic identity of every node (left brow <-> right brow, and so on).
    """
    pairs_68 = [
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        (31, 35), (32, 34),
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
        (48, 54), (49, 53), (50, 52), (55, 59), (56, 58),
        (60, 64), (61, 63), (65, 67),
    ]
    perm = np.arange(51)
    for a, b in pairs_68:
        perm[a - _FACE_OFFSET] = b - _FACE_OFFSET
        perm[b - _FACE_OFFSET] = a - _FACE_OFFSET
    return perm


# --------------------------------------------------------------------------- #
# EEG channel graph
# --------------------------------------------------------------------------- #

# Schematic 2-D positions of the 10-20 electrodes on the unit disc.
_ELECTRODE_XY: Dict[str, Tuple[float, float]] = {
    "FP1": (-0.31, 0.95), "FP2": (0.31, 0.95),
    "F7": (-0.81, 0.59), "F3": (-0.40, 0.50), "FZ": (0.00, 0.50), "F4": (0.40, 0.50), "F8": (0.81, 0.59),
    "T7": (-1.00, 0.00), "C3": (-0.50, 0.00), "CZ": (0.00, 0.00), "C4": (0.50, 0.00), "T8": (1.00, 0.00),
    "P7": (-0.81, -0.59), "P3": (-0.40, -0.50), "PZ": (0.00, -0.50), "P4": (0.40, -0.50), "P8": (0.81, -0.59),
    "O1": (-0.31, -0.95), "O2": (0.31, -0.95),
}

CHBMIT_CHANNELS: List[str] = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

SIENA_CHANNELS: List[str] = CHBMIT_CHANNELS[:16]


def bipolar_positions(channels: Sequence[str]) -> np.ndarray:
    """Midpoint of the two electrodes of every bipolar channel, shape ``[N, 2]``."""
    out = np.zeros((len(channels), 2), dtype=np.float32)
    for k, name in enumerate(channels):
        first, second = name.upper().split("-")
        out[k] = 0.5 * (np.array(_ELECTRODE_XY[first]) + np.array(_ELECTRODE_XY[second]))
    return out


def eeg_graph(channels: Sequence[str], threshold: float = 0.5) -> np.ndarray:
    """Distance-threshold adjacency of bipolar EEG channels.

    Pairwise Euclidean distances between channel midpoints are normalised to
    [0, 1] by the largest distance; channels closer than ``threshold`` are
    connected.
    """
    pos = bipolar_positions(channels)
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    dist = dist / max(float(dist.max()), 1e-12)
    adj = (dist <= threshold).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


# --------------------------------------------------------------------------- #
# Skeleton graph (NTU RGB+D, 25 joints)
# --------------------------------------------------------------------------- #

# Bone list in the 1-based NTU joint numbering (child, parent), rooted at
# joint 2 (middle of the spine), which is also the reference joint used to
# normalise the coordinates.
_NTU_BONES_1BASED: List[Tuple[int, int]] = [
    (1, 2), (21, 2), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
    (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
    (17, 1), (18, 17), (19, 18), (20, 19), (22, 8), (23, 8), (24, 12), (25, 12),
]
NTU_NUM_JOINTS = 25
NTU_ROOT_JOINT = 1  # zero-based index of joint 2 (middle of the spine)


def skeleton_graph() -> np.ndarray:
    """Binary adjacency of the 25-joint NTU skeleton (24 bones)."""
    edges = [(c - 1, p - 1) for c, p in _NTU_BONES_1BASED]
    return edges_to_adjacency(NTU_NUM_JOINTS, edges)


def skeleton_parents() -> np.ndarray:
    """Parent index of every joint (the root points to itself)."""
    parents = np.arange(NTU_NUM_JOINTS)
    for child, parent in _NTU_BONES_1BASED:
        parents[child - 1] = parent - 1
    return parents


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def build_graph(name: str) -> np.ndarray:
    """Return the predefined adjacency for ``face``, ``eeg_chbmit``, ``eeg_siena`` or ``skeleton``."""
    name = name.lower()
    if name == "face":
        return face_graph()
    if name in ("eeg_chbmit", "chbmit"):
        return eeg_graph(CHBMIT_CHANNELS)
    if name in ("eeg_siena", "siena"):
        return eeg_graph(SIENA_CHANNELS)
    if name in ("skeleton", "ntu", "ntu60", "ntu120"):
        return skeleton_graph()
    raise ValueError(f"Unknown graph '{name}'")


def graph_name_for_task(task: str, dataset: str) -> str:
    if task == "fer":
        return "face"
    if task == "esd":
        return f"eeg_{dataset}"
    if task == "har":
        return "skeleton"
    raise ValueError(f"Unknown task '{task}'")
