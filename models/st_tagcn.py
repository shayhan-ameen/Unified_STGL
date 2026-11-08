"""Spatio-Temporal Transition-Adaptive Graph Convolutional Network (ST-TAGCN).

ST-TAGCN learns the final representation over the dynamic graph
G_D(X, A_D) reconstructed by ST-DEGaD and decides *which propagated
information should receive greater emphasis*.  Each block performs

    SGC(A_D) -> BN -> ReLU -> TASM -> TConv -> BN -> ReLU -> TATM -> + residual

where the two modulation modules derive their weights from directional
feature transitions between consecutive temporal positions:

* ``transition_scores``  Delta^t_i = 1 - cos(F^t_i, F^{t-1}_i) (or the L1 / L2
                         alternatives of the ablation study), with the first
                         transition replicated to position 1.
* ``TASM``               alpha_SP = moving average of Delta over a temporal
                         neighbourhood of radius r_mod; F + F * alpha_SP.
* ``TATM``               alpha_TM = mean of Delta over nodes; F + F * alpha_TM.

Both are parameter-free.  The ablation alternatives (SE, CBAM, spatio-temporal
self-attention, static learnable weights) and the plain ST-GCN block used when
ST-TAGCN is switched off are implemented here as well so that every variant
shares the same interface: ``module(F, delta) -> (F', alpha)``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ChannelNorm, ResidualProjection, SpatialGraphConv, TemporalConv


# --------------------------------------------------------------------------- #
# Transition measures
# --------------------------------------------------------------------------- #


def transition_scores(features: torch.Tensor, measure: str = "cosine", eps: float = 1e-8) -> torch.Tensor:
    """Per-node transition scores between consecutive temporal positions.

    ``features``: ``[B, T, N, C]`` -> ``[B, T, N]`` with values in [0, 1].
    The first position repeats the first available transition.  L1 and L2
    magnitudes are scaled by their per-sample maximum so that they share the
    [0, 1] range of the cosine distance.
    """
    current, previous = features[:, 1:], features[:, :-1]
    if measure == "cosine":
        delta = 1.0 - F.cosine_similarity(current, previous, dim=-1, eps=eps)
    elif measure == "l1":
        delta = (current - previous).abs().sum(dim=-1)
        delta = delta / delta.amax(dim=(1, 2), keepdim=True).clamp(min=eps)
    elif measure == "l2":
        delta = (current - previous).norm(dim=-1)
        delta = delta / delta.amax(dim=(1, 2), keepdim=True).clamp(min=eps)
    else:
        raise ValueError(f"Unknown feature transition measure '{measure}'")
    delta = delta.clamp(0.0, 1.0)
    return torch.cat([delta[:, :1], delta], dim=1)


def local_temporal_average(delta: torch.Tensor, radius: int) -> torch.Tensor:
    """Average ``[B, T, N]`` scores over a window of 2r + 1 positions with replicate padding."""
    if radius == 0:
        return delta
    batch, length, nodes = delta.shape
    seq = delta.permute(0, 2, 1).reshape(batch * nodes, 1, length)
    seq = F.pad(seq, (radius, radius), mode="replicate")
    seq = F.avg_pool1d(seq, kernel_size=2 * radius + 1, stride=1)
    return seq.reshape(batch, nodes, length).permute(0, 2, 1)


# --------------------------------------------------------------------------- #
# Transition-adaptive modulation (proposed)
# --------------------------------------------------------------------------- #


class TASM(nn.Module):
    """Transition-Adaptive Spatial Modulation: node weights from local transition strength."""

    def __init__(self, radius: int = 1) -> None:
        super().__init__()
        self.radius = radius

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = local_temporal_average(delta, self.radius)       # [B, T, N]
        return x + x * alpha.unsqueeze(-1), alpha


class TATM(nn.Module):
    """Transition-Adaptive Temporal Modulation: position weights from the node-averaged transition."""

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = delta.mean(dim=2)                                # [B, T]
        return x + x * alpha[:, :, None, None], alpha


# --------------------------------------------------------------------------- #
# Ablation alternatives
# --------------------------------------------------------------------------- #


class SEModulation(nn.Module):
    """Squeeze-and-excitation channel attention placed at the TASM/TATM location."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.fc = nn.Sequential(nn.Linear(channels, reduced), nn.ReLU(inplace=True), nn.Linear(reduced, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = self.fc(x.mean(dim=(1, 2)))[:, None, None, :]
        return x * weight, weight.mean(dim=-1).expand(x.shape[0], x.shape[1], x.shape[2])


class CBAMModulation(nn.Module):
    """CBAM: channel attention followed by spatial attention over the (T, N) map."""

    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 7) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.mlp = nn.Sequential(nn.Linear(channels, reduced), nn.ReLU(inplace=True), nn.Linear(reduced, channels))
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        channel = torch.sigmoid(self.mlp(x.mean(dim=(1, 2))) + self.mlp(x.amax(dim=2).amax(dim=1)))
        x = x * channel[:, None, None, :]
        descriptor = torch.stack([x.mean(dim=-1), x.amax(dim=-1)], dim=1)   # [B, 2, T, N]
        spatial = torch.sigmoid(self.spatial(descriptor)).squeeze(1)       # [B, T, N]
        return x * spatial.unsqueeze(-1), spatial


class GraphSelfAttention(nn.Module):
    """GAT-style attention over node tokens at every temporal position (replaces TASM)."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(channels, channels, bias=False)
        self.attn_src = nn.Linear(channels, 1, bias=False)
        self.attn_dst = nn.Linear(channels, 1, bias=False)
        self.out = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        scores = F.leaky_relu(self.attn_src(h) + self.attn_dst(h).transpose(2, 3), 0.2)   # [B, T, N, N]
        weights = self.dropout(scores.softmax(dim=-1))
        out = self.dropout(self.out(torch.einsum("btij,btjc->btic", weights, h)))
        return x + out, weights.mean(dim=-1)


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention over temporal tokens of every node (replaces TATM)."""

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length, nodes, channels = x.shape
        tokens = x.permute(0, 2, 1, 3).reshape(batch * nodes, length, channels)
        out, weights = self.attn(tokens, tokens, tokens, need_weights=True)
        out = self.dropout(self.out(out)).reshape(batch, nodes, length, channels).permute(0, 2, 1, 3)
        alpha = weights.mean(dim=1).reshape(batch, nodes, length).mean(dim=1)   # [B, T]
        return x + out, alpha


class StaticModulation(nn.Module):
    """Trainable, sample-independent weights at the TASM or TATM location.

    The weights are initialised from the transition statistics of the first
    training batch and then learned like any other parameter.
    """

    def __init__(self, length: int, nodes: int, spatial: bool) -> None:
        super().__init__()
        shape = (1, length, nodes, 1) if spatial else (1, length, 1, 1)
        self.spatial = spatial
        self.alpha = nn.Parameter(torch.full(shape, 0.5))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and not bool(self.initialized):
            with torch.no_grad():
                stats = delta.mean(dim=0, keepdim=True).unsqueeze(-1)       # [1, T, N, 1]
                if not self.spatial:
                    stats = stats.mean(dim=2, keepdim=True)
                self.alpha.copy_(stats.expand_as(self.alpha))
                self.initialized.fill_(True)
        alpha = self.alpha.expand(x.shape[0], -1, -1, -1)
        return x + x * alpha, (alpha.squeeze(-1) if self.spatial else alpha[:, :, 0, 0])


def build_modulation(kind: str, channels: int, spatial: bool, radius: int, length: int, nodes: int) -> nn.Module:
    """Factory for the modulation at the TASM (``spatial=True``) or TATM position."""
    kind = kind.lower()
    if kind == "transition":
        return TASM(radius) if spatial else TATM()
    if kind == "se":
        return SEModulation(channels)
    if kind == "cbam":
        return CBAMModulation(channels)
    if kind == "st_selfattn":
        return GraphSelfAttention(channels) if spatial else TemporalSelfAttention(channels)
    if kind == "static_learnable":
        return StaticModulation(length, nodes, spatial)
    raise ValueError(f"Unknown modulation '{kind}'")


# --------------------------------------------------------------------------- #
# Blocks and network
# --------------------------------------------------------------------------- #


class STTAGCNBlock(nn.Module):
    """One ST-TAGCN block (Fig. "ST-TAGCN block" in the paper)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 3,
        mod_radius: int = 1,
        transition_measure: str = "cosine",
        modulation: str = "transition",
        cos_eps: float = 1e-8,
        sg_reduction: int = 2,
        sg_min_width: int = 16,
        length: int = 8,
        nodes: int = 51,
    ) -> None:
        super().__init__()
        self.transition_measure = transition_measure
        self.cos_eps = cos_eps
        self.gcn = SpatialGraphConv(in_channels, out_channels, sg_reduction, sg_min_width)
        self.gcn_norm = ChannelNorm(out_channels)
        self.tasm = build_modulation(modulation, out_channels, True, mod_radius, length, nodes)
        self.tconv = TemporalConv(out_channels, out_channels, temporal_kernel, sg_reduction, sg_min_width)
        self.tconv_norm = ChannelNorm(out_channels)
        self.tatm = build_modulation(modulation, out_channels, False, mod_radius, length, nodes)
        self.residual = ResidualProjection(in_channels, out_channels)

    def _delta(self, features: torch.Tensor, prior: Optional[torch.Tensor]) -> torch.Tensor:
        if self.transition_measure in ("flow_mag", "landmark_vel"):
            if prior is None:
                raise ValueError(f"transition measure '{self.transition_measure}' needs an input-level prior")
            return prior
        return transition_scores(features, self.transition_measure, self.cos_eps)

    def forward(
        self, y: torch.Tensor, adj_dynamic: torch.Tensor, prior: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``y``: ``[B, T, N, C_k]``, ``adj_dynamic``: ``[B, N, N]`` -> ``[B, T, N, C_{k+1}]``, alpha_SP, alpha_TM."""
        f_sp = F.relu(self.gcn_norm(self.gcn(y, adj_dynamic)))
        f_sp, alpha_sp = self.tasm(f_sp, self._delta(f_sp, prior))
        f_tm = F.relu(self.tconv_norm(self.tconv(f_sp)))
        f_tm, alpha_tm = self.tatm(f_tm, self._delta(f_tm, prior))
        return f_tm + self.residual(y), alpha_sp, alpha_tm


class STGCNBlock(nn.Module):
    """Conventional ST-GCN block without transition-adaptive modulation (ablation)."""

    def __init__(
        self, in_channels: int, out_channels: int, temporal_kernel: int = 3, sg_reduction: int = 2, sg_min_width: int = 16
    ) -> None:
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, sg_reduction, sg_min_width)
        self.gcn_norm = ChannelNorm(out_channels)
        self.tconv = TemporalConv(out_channels, out_channels, temporal_kernel, sg_reduction, sg_min_width)
        self.tconv_norm = ChannelNorm(out_channels)
        self.residual = ResidualProjection(in_channels, out_channels)

    def forward(
        self, y: torch.Tensor, adj_dynamic: torch.Tensor, prior: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.relu(self.gcn_norm(self.gcn(y, adj_dynamic)))
        h = self.tconv_norm(self.tconv(h))
        out = F.relu(h + self.residual(y))
        ones_sp = out.new_ones(out.shape[0], out.shape[1], out.shape[2])
        return out, ones_sp, ones_sp.mean(dim=2)


class STTAGCN(nn.Module):
    """K stacked blocks sharing the dynamic adjacency A_D."""

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        temporal_kernel: int = 3,
        mod_radius: int = 1,
        transition_measure: str = "cosine",
        modulation: str = "transition",
        cos_eps: float = 1e-8,
        sg_reduction: int = 2,
        sg_min_width: int = 16,
        length: int = 8,
        nodes: int = 51,
        transition_adaptive: bool = True,
    ) -> None:
        super().__init__()
        blocks = []
        prev = in_channels
        for width in channels:
            if transition_adaptive:
                blocks.append(
                    STTAGCNBlock(
                        prev, width, temporal_kernel, mod_radius, transition_measure, modulation, cos_eps,
                        sg_reduction, sg_min_width, length, nodes,
                    )
                )
            else:
                blocks.append(STGCNBlock(prev, width, temporal_kernel, sg_reduction, sg_min_width))
            prev = width
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = prev

    def forward(
        self, x: torch.Tensor, adj_dynamic: torch.Tensor, prior: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        alphas_sp, alphas_tm = [], []
        for block in self.blocks:
            x, alpha_sp, alpha_tm = block(x, adj_dynamic, prior)
            alphas_sp.append(alpha_sp)
            alphas_tm.append(alpha_tm)
        return x, {"alpha_spatial": alphas_sp, "alpha_temporal": alphas_tm}
