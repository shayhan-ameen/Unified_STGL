"""Spatio-Temporal Decoupled Attention (ST-DA) and the attention variants
used in the ablation study of the Structural Encoder.

ST-DA (paper Sec. "Spatio-Temporal Decoupled Attention")
    1. Temporal descriptors by average and max pooling over nodes, spatial
       descriptors by average and max pooling over temporal positions.
    2. Hybrid Pooling Fusion: a linear map from the concatenated [avg || max]
       descriptors back to C channels, separately for the two axes.
    3. Position-wise concatenation of the N spatial and T temporal descriptors,
       a shared bottleneck (linear C -> C / r, batch norm, ReLU).
    4. Two sigmoid heads produce the spatial coefficients beta_SP in R^{N x C}
       and temporal coefficients beta_TM in R^{T x C}; their broadcast product
       is the joint map beta_ST in R^{T x N x C}.
    5. The block output is F * beta_ST.

Every module here maps ``[B, T, N, C] -> ([B, T, N, C], attention tensor)``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class STDecoupledAttention(nn.Module):
    """ST-DA with Hybrid Pooling Fusion."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.hpf_temporal = nn.Linear(2 * channels, channels)
        self.hpf_spatial = nn.Linear(2 * channels, channels)
        self.bottleneck = nn.Linear(channels, reduced)
        self.bottleneck_bn = nn.BatchNorm1d(reduced)
        self.head_spatial = nn.Linear(reduced, channels, bias=False)
        self.head_temporal = nn.Linear(reduced, channels, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length, nodes, _ = x.shape
        # Descriptors.
        temporal = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=-1)   # [B, T, 2C]
        spatial = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)    # [B, N, 2C]
        temporal = self.hpf_temporal(temporal)                          # [B, T, C]
        spatial = self.hpf_spatial(spatial)                             # [B, N, C]
        # Shared bottleneck over the (N + T) positions.
        joint = torch.cat([spatial, temporal], dim=1)                   # [B, N + T, C]
        joint = self.bottleneck(joint)                                  # [B, N + T, C_rd]
        joint = self.bottleneck_bn(joint.transpose(1, 2)).transpose(1, 2)
        joint = torch.relu(joint)
        spatial_rd, temporal_rd = torch.split(joint, [nodes, length], dim=1)
        # Attention coefficients.
        beta_spatial = torch.sigmoid(self.head_spatial(spatial_rd))     # [B, N, C]
        beta_temporal = torch.sigmoid(self.head_temporal(temporal_rd))  # [B, T, C]
        beta = beta_temporal.unsqueeze(2) * beta_spatial.unsqueeze(1)   # [B, T, N, C]
        return x * beta, beta


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention (ablation: 'channel')."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = self.fc(x.mean(dim=(1, 2)))                            # [B, C]
        weight = weight[:, None, None, :]
        return x * weight, weight.expand_as(x)


class FrameAttention(nn.Module):
    """Frame (temporal-position) attention (ablation: 'frame').

    Average and max pooling over channels and nodes give two descriptors per
    temporal position; a 1-D convolution over time followed by a sigmoid
    yields one weight per position.
    """

    def __init__(self, channels: int, kernel_size: int = 9) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        avg = x.mean(dim=(2, 3))                                        # [B, T]
        mx = x.amax(dim=3).amax(dim=2)                                  # [B, T]
        weight = torch.sigmoid(self.conv(torch.stack([avg, mx], dim=1)))  # [B, 1, T]
        weight = weight.transpose(1, 2)[:, :, :, None]                  # [B, T, 1, 1]
        return x * weight, weight.expand_as(x)


class NoAttention(nn.Module):
    """Identity (ablation: 'none')."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, torch.ones_like(x)


def build_attention(kind: str, channels: int, reduction: int = 4) -> nn.Module:
    """Factory for the Structural Encoder attention: none | channel | frame | stda."""
    kind = kind.lower()
    if kind == "stda":
        return STDecoupledAttention(channels, reduction)
    if kind == "channel":
        return ChannelAttention(channels, reduction)
    if kind == "frame":
        return FrameAttention(channels)
    if kind == "none":
        return NoAttention()
    raise ValueError(f"Unknown attention type '{kind}'")
