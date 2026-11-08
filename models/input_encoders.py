"""Task-specific input construction mapping raw node inputs to the common
projected feature tensor X in R^{B x T x N x F'}.

* ``FaceNodeEncoder``      local optical-flow magnitude patches
                           ``[B, T, N, Sz, Sz]`` -> depthwise 3x3 convolution with
                           one group per landmark -> flatten -> linear Sz^2 -> F'.
* ``EEGNodeEncoder``       window signals ``[B, T, N, n_w]`` -> shared dilated causal
                           TCN -> average pooling over the window -> linear F_EEG -> F'.
* ``SkeletonNodeEncoder``  joint coordinates or bone displacements
                           ``[B, T, N, 3]`` -> linear 3 -> F'.

The final linear map of every encoder is the modality-specific input projection
(W_proj, b_proj) of the paper.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceNodeEncoder(nn.Module):
    """Depthwise convolution over landmark motion patches followed by projection."""

    def __init__(self, num_nodes: int, patch_size: int, out_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.patch_size = patch_size
        self.dwconv = nn.Conv2d(num_nodes, num_nodes, kernel_size, padding=kernel_size // 2, groups=num_nodes)
        self.proj = nn.Linear(patch_size * patch_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, T, N, Sz, Sz]`` -> ``[B, T, N, F']``."""
        batch, length, nodes, size, _ = x.shape
        patches = self.dwconv(x.reshape(batch * length, nodes, size, size))
        return self.proj(patches.reshape(batch, length, nodes, size * size))


class CausalConv1d(nn.Module):
    """1-D convolution whose output at time t depends only on inputs at times <= t."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNLayer(nn.Module):
    """Dilated causal convolution with ReLU and a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x)) + self.skip(x)


class EEGNodeEncoder(nn.Module):
    """Shared temporal convolutional network producing one feature per channel and window."""

    def __init__(
        self, out_dim: int, channels: int = 64, kernel_size: int = 3, dilations: List[int] = (1, 2, 4, 8)
    ) -> None:
        super().__init__()
        layers = []
        prev = 1
        for dilation in dilations:
            layers.append(TCNLayer(prev, channels, kernel_size, dilation))
            prev = channels
        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Linear(channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, T, N, n_w]`` -> ``[B, T, N, F']``."""
        batch, length, nodes, samples = x.shape
        signal = x.reshape(batch * length * nodes, 1, samples)
        hidden = self.tcn(signal).mean(dim=-1)                   # global average pooling over the window
        return self.proj(hidden.reshape(batch, length, nodes, -1))


class SkeletonNodeEncoder(nn.Module):
    """Linear projection of 3-D joint coordinates or bone displacements."""

    def __init__(self, out_dim: int, in_dim: int = 3) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, T, N, 3]`` -> ``[B, T, N, F']``."""
        return self.proj(x)


def build_input_encoder(cfg) -> nn.Module:
    """Create the encoder named by ``cfg.model.encoder`` (face | eeg | skeleton)."""
    kind = cfg.model.encoder.lower()
    out_dim = cfg.model.in_dim
    if kind == "face":
        return FaceNodeEncoder(cfg.data.num_nodes, cfg.data.patch_size, out_dim)
    if kind == "eeg":
        enc = cfg.model.get("eeg_encoder", {})
        return EEGNodeEncoder(
            out_dim,
            channels=enc.get("channels", 64),
            kernel_size=enc.get("kernel_size", 3),
            dilations=list(enc.get("dilations", [1, 2, 4, 8])),
        )
    if kind == "skeleton":
        return SkeletonNodeEncoder(out_dim)
    raise ValueError(f"Unknown input encoder '{kind}'")
