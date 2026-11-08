"""Spatio-Temporal Dual-Encoder Gated Decoder (ST-DEGaD).

ST-DEGaD reconstructs a sample-specific, sequence-level dynamic adjacency
matrix A_D from the projected node features X in R^{T x N x F'} and the
predefined adjacency A.  It decides *where information should propagate*.

Components
----------
``STDAGCNBlock`` / ``StructuralEncoder``
    Space-first pathway: graph convolution over the normalised predefined
    topology, temporal convolution with a residual + layer norm, and
    Spatio-Temporal Decoupled Attention.  L stacked blocks give Z_S.
``ContextualEncoder``
    Time-first pathway: feature projection, Hydra bidirectional mixing along
    each node trajectory (residual + layer norm), and global spatial
    self-attention among nodes at every temporal position (residual + layer
    norm).  It never sees the predefined topology and gives Z_C.
``GatedDecoder``
    Cross-spatio-temporal relation modelling within a temporal radius delta,
    dual-encoder gating, edge-specific temporal-pair fusion, and adjacency
    refinement (symmetrisation, self-connections, symmetric normalisation).
``STDEGaD``
    The three components together; ``forward(X, A)`` returns ``A_D`` with the
    intermediate encoder outputs and attention maps for inspection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .hydra import Hydra
from .layers import ChannelNorm, SpatialGraphConv, TemporalConv
from .stda import build_attention


# --------------------------------------------------------------------------- #
# Structural Encoder
# --------------------------------------------------------------------------- #


class STDAGCNBlock(nn.Module):
    """One Spatio-Temporal Decoupled-Attention GCN block: GCN -> TConv -> ST-DA."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 3,
        attention: str = "stda",
        reduction: int = 4,
        sg_reduction: int = 2,
        sg_min_width: int = 16,
    ) -> None:
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, sg_reduction, sg_min_width)
        self.gcn_norm = ChannelNorm(out_channels)
        self.tconv = TemporalConv(out_channels, out_channels, temporal_kernel, sg_reduction, sg_min_width)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.attention = build_attention(attention, out_channels, reduction)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``x``: ``[B, T, N, C_l]``, ``adj_norm``: ``[N, N]`` -> ``[B, T, N, C_{l+1}]`` and the attention map."""
        h = F.relu(self.gcn_norm(self.gcn(x, adj_norm)))      # local spatial propagation
        u = self.tconv(h)                                      # short-term temporal evolution
        f = self.layer_norm(h + u)                             # residual + LN
        return self.attention(f)


class StructuralEncoder(nn.Module):
    """L stacked ST-DAGCN blocks sharing the normalised predefined adjacency."""

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        temporal_kernel: int = 3,
        attention: str = "stda",
        reduction: int = 4,
        sg_reduction: int = 2,
        sg_min_width: int = 16,
    ) -> None:
        super().__init__()
        blocks = []
        prev = in_channels
        for width in channels:
            blocks.append(
                STDAGCNBlock(prev, width, temporal_kernel, attention, reduction, sg_reduction, sg_min_width)
            )
            prev = width
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = prev

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attentions = []
        for block in self.blocks:
            x, att = block(x, adj_norm)
            attentions.append(att)
        return x, attentions


# --------------------------------------------------------------------------- #
# Contextual Encoder
# --------------------------------------------------------------------------- #


class GlobalSpatialAttention(nn.Module):
    """Single-head scaled dot-product attention among the N nodes of each temporal position."""

    def __init__(self, dim: int, attention_dim: int) -> None:
        super().__init__()
        self.scale = attention_dim ** -0.5
        self.query = nn.Linear(dim, attention_dim, bias=False)
        self.key = nn.Linear(dim, attention_dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``h``: ``[B, T, N, C]`` -> aggregated ``[B, T, N, C]`` and relations ``[B, T, N, N]``."""
        scores = torch.einsum("btid,btjd->btij", self.query(h), self.key(h)) * self.scale
        relation = scores.softmax(dim=-1)
        out = torch.einsum("btij,btjc->btic", relation, self.value(h))
        return self.out(out), relation


class ContextualEncoder(nn.Module):
    """Time-first encoder: Hydra along node trajectories, then global spatial attention."""

    def __init__(
        self,
        in_channels: int,
        dim: int = 128,
        attention_dim: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, dim, bias=False)
        self.hydra = Hydra(dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.norm_temporal = nn.LayerNorm(dim)
        self.spatial = GlobalSpatialAttention(dim, attention_dim)
        self.norm_spatial = nn.LayerNorm(dim)
        self.out_channels = dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``x``: ``[B, T, N, F']`` -> ``Z_C``: ``[B, T, N, C_L]`` and spatial relations ``[B, T, N, N]``."""
        batch, length, nodes, _ = x.shape
        h = self.proj(x)
        # Long-range temporal modelling, independently per node.
        trajectories = h.permute(0, 2, 1, 3).reshape(batch * nodes, length, -1)
        mixed = self.hydra(trajectories).reshape(batch, nodes, length, -1).permute(0, 2, 1, 3)
        h = self.norm_temporal(h + mixed)
        # Global spatial context modelling at every temporal position.
        out, relation = self.spatial(h)
        return self.norm_spatial(h + out), relation


# --------------------------------------------------------------------------- #
# Gated Decoder
# --------------------------------------------------------------------------- #


def pair_features(z_s: torch.Tensor, z_c: torch.Tensor, offset: int) -> torch.Tensor:
    """Pair representation e_ij^{t,t'} for t' = t + offset.

    ``z_s``, ``z_c``: ``[B, T, N, C]``.  Returns ``[B, T_valid, N, N, 4C]``
    where ``T_valid`` is the number of positions t for which t + offset is a
    valid temporal position.
    """
    length = z_s.shape[1]
    lo, hi = max(0, -offset), min(length, length - offset)
    zs_i, zs_j = z_s[:, lo:hi, :, None, :], z_s[:, lo + offset : hi + offset, None, :, :]
    zc_i, zc_j = z_c[:, lo:hi, :, None, :], z_c[:, lo + offset : hi + offset, None, :, :]
    return torch.cat([zs_i * zs_j, (zs_i - zs_j).abs(), zc_i * zc_j, (zc_i - zc_j).abs()], dim=-1)


def relation_scores(r: torch.Tensor, offset: int) -> torch.Tensor:
    """Scaled dot-product relation scores between positions t and t + offset.

    ``r``: ``[B, T, N, d_rel]`` -> ``[B, T_valid, N, N]``.
    """
    length = r.shape[1]
    lo, hi = max(0, -offset), min(length, length - offset)
    scale = r.shape[-1] ** -0.5
    return torch.einsum("btid,btjd->btij", r[:, lo:hi], r[:, lo + offset : hi + offset]) * scale


class GatedDecoder(nn.Module):
    """Reconstruct the sample-specific dynamic adjacency A_D from Z_S and Z_C.

    Parameters
    ----------
    channels    : encoder width C_L.
    radius      : temporal decoding radius delta.
    rel_dim     : relation-projection dimension d_rel.
    gate_hidden : hidden width of the gating network Gamma.
    pair_hidden : hidden width d_pair of the temporal-pair scoring network.
    use_checkpoint : recompute pair features during backward to save memory.
    """

    def __init__(
        self,
        channels: int,
        radius: int = 1,
        rel_dim: int = 32,
        gate_hidden: int = 64,
        pair_hidden: int = 64,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.use_checkpoint = use_checkpoint
        self.rel_structural = nn.Linear(channels, rel_dim, bias=False)
        self.rel_contextual = nn.Linear(channels, rel_dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(4 * channels, gate_hidden), nn.ReLU(inplace=True), nn.Linear(gate_hidden, 2))
        self.pair_score = nn.Sequential(nn.Linear(4 * channels, pair_hidden), nn.Tanh(), nn.Linear(pair_hidden, 1, bias=False))

    def _offset_relations(
        self, z_s: torch.Tensor, z_c: torch.Tensor, r_s: torch.Tensor, r_c: torch.Tensor, offset: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gated relation strengths rho and pair logits xi for one temporal offset."""
        e = pair_features(z_s, z_c, offset)                      # [B, T_v, N, N, 4C]
        gamma = self.gate(e).softmax(dim=-1)                     # [B, T_v, N, N, 2]
        s_s = relation_scores(r_s, offset)                       # [B, T_v, N, N]
        s_c = relation_scores(r_c, offset)
        rho = torch.sigmoid(gamma[..., 0] * s_s + gamma[..., 1] * s_c)
        xi = self.pair_score(e).squeeze(-1)                      # [B, T_v, N, N]
        return rho, xi

    def forward(self, z_s: torch.Tensor, z_c: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """``z_s``, ``z_c``: ``[B, T, N, C_L]`` -> ``A_D``: ``[B, N, N]`` plus intermediates."""
        r_s, r_c = self.rel_structural(z_s), self.rel_contextual(z_c)
        rhos, xis = [], []
        for offset in range(-self.radius, self.radius + 1):
            if self.use_checkpoint and torch.is_grad_enabled():
                rho, xi = checkpoint.checkpoint(
                    self._offset_relations, z_s, z_c, r_s, r_c, offset, use_reentrant=False
                )
            else:
                rho, xi = self._offset_relations(z_s, z_c, r_s, r_c, offset)
            rhos.append(rho)
            xis.append(xi)
        rho = torch.cat(rhos, dim=1)                             # [B, |Omega|, N, N]
        pi = torch.cat(xis, dim=1).softmax(dim=1)                # edge-specific temporal-pair weights
        rho_bar = (pi * rho).sum(dim=1)                          # [B, N, N] sequence-level relations

        # Adjacency refinement.
        rho_sym = 0.5 * (rho_bar + rho_bar.transpose(1, 2))
        eye = torch.eye(rho_sym.shape[-1], device=rho_sym.device, dtype=rho_sym.dtype)
        adj = rho_sym * (1.0 - eye) + eye
        inv_sqrt = adj.sum(dim=-1).clamp(min=1e-12).rsqrt()
        adj_dynamic = inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)
        return adj_dynamic, {"relation": rho_bar, "pair_weights": pi}


# --------------------------------------------------------------------------- #
# Full ST-DEGaD
# --------------------------------------------------------------------------- #


class STDEGaD(nn.Module):
    """Dual Encoder + Gated Decoder producing the sample-specific adjacency A_D."""

    def __init__(
        self,
        in_channels: int,
        structural_channels: List[int],
        contextual_dim: int = 128,
        temporal_kernel: int = 3,
        attention: str = "stda",
        reduction: int = 4,
        attention_dim: int = 32,
        hydra_d_state: int = 16,
        hydra_d_conv: int = 4,
        hydra_expand: int = 2,
        hydra_headdim: int = 32,
        radius: int = 1,
        rel_dim: int = 32,
        gate_hidden: int = 64,
        pair_hidden: int = 64,
        sg_reduction: int = 2,
        sg_min_width: int = 16,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if structural_channels[-1] != contextual_dim:
            raise ValueError("The last Structural Encoder width must equal the Contextual Encoder width C_L")
        self.structural = StructuralEncoder(
            in_channels, structural_channels, temporal_kernel, attention, reduction, sg_reduction, sg_min_width
        )
        self.contextual = ContextualEncoder(
            in_channels, contextual_dim, attention_dim, hydra_d_state, hydra_d_conv, hydra_expand, hydra_headdim
        )
        self.decoder = GatedDecoder(contextual_dim, radius, rel_dim, gate_hidden, pair_hidden, use_checkpoint)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """``x``: ``[B, T, N, F']``, ``adj_norm``: normalised predefined ``[N, N]`` -> ``A_D`` ``[B, N, N]``."""
        z_s, stda_maps = self.structural(x, adj_norm)
        z_c, spatial_relation = self.contextual(x)
        adj_dynamic, decoder_aux = self.decoder(z_s, z_c)
        aux = {
            "z_structural": z_s,
            "z_contextual": z_c,
            "stda": stda_maps,
            "contextual_relation": spatial_relation,
            **decoder_aux,
        }
        return adj_dynamic, aux
