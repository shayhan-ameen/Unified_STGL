"""Building blocks shared by the Structural Encoder and ST-TAGCN.

All modules take and return tensors in the paper's ``[B, T, N, C]`` layout
(batch, temporal position, graph node, channel) and permute internally to the
``[B, C, T, N]`` layout expected by 2-D convolutions.

* ``SGLayer``           separable bottleneck of EfficientGCN: depthwise (k x 1)
                        -> pointwise C_in -> C_b -> pointwise C_b -> C_out ->
                        depthwise (k x 1), with C_b = max(C_out / r, min_width).
* ``SpatialGraphConv``  A X W + b realised as graph aggregation followed by an
                        SGLayer with k = 1.
* ``TemporalConv``      TConv_kappa realised as an SGLayer with k = kappa and
                        "same" padding so T is preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def to_conv_layout(x: torch.Tensor) -> torch.Tensor:
    """``[B, T, N, C]`` -> ``[B, C, T, N]``."""
    return x.permute(0, 3, 1, 2).contiguous()


def to_graph_layout(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, T, N]`` -> ``[B, T, N, C]``."""
    return x.permute(0, 2, 3, 1).contiguous()


class SGLayer(nn.Module):
    """Separable bottleneck layer (EfficientGCN's SGLayer).

    Parameters
    ----------
    in_channels, out_channels : channel widths.
    kernel_size : temporal kernel; ``1`` for the spatial transform.
    reduction : bottleneck reduction ratio r_sg.
    min_width : lower bound on the bottleneck width.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        reduction: int = 2,
        min_width: int = 16,
    ) -> None:
        super().__init__()
        inner = max(out_channels // reduction, min_width)
        pad = (kernel_size - 1) // 2
        self.depth_in = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (kernel_size, 1), padding=(pad, 0), groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.point_in = nn.Sequential(
            nn.Conv2d(in_channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
        )
        self.point_out = nn.Sequential(
            nn.Conv2d(inner, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.depth_out = nn.Conv2d(
            out_channels, out_channels, (kernel_size, 1), padding=(pad, 0), groups=out_channels, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, C_in, T, N]`` -> ``[B, C_out, T, N]``."""
        x = self.depth_in(x)
        x = self.point_in(x)
        x = self.point_out(x)
        return self.depth_out(x)


class SpatialGraphConv(nn.Module):
    """Graph convolution ``A X W + b`` applied independently at every temporal position.

    The adjacency may be shared by the batch (``[N, N]``) or sample-specific
    (``[B, N, N]``); it is used as given, so any normalisation must be applied
    beforehand.
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 2, min_width: int = 16) -> None:
        super().__init__()
        self.transform = SGLayer(in_channels, out_channels, kernel_size=1, reduction=reduction, min_width=min_width)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, T, N, C_in]``, ``adj``: ``[N, N]`` or ``[B, N, N]`` -> ``[B, T, N, C_out]``."""
        if adj.dim() == 2:
            aggregated = torch.einsum("ij,btjc->btic", adj, x)
        else:
            aggregated = torch.einsum("bij,btjc->btic", adj, x)
        out = self.transform(to_conv_layout(aggregated))
        return to_graph_layout(out)


class TemporalConv(nn.Module):
    """Temporal convolution ``TConv_kappa`` preserving T, N, and (optionally changing) C."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, reduction: int = 2, min_width: int = 16
    ) -> None:
        super().__init__()
        self.transform = SGLayer(in_channels, out_channels, kernel_size=kernel_size, reduction=reduction, min_width=min_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, T, N, C_in]`` -> ``[B, T, N, C_out]``."""
        return to_graph_layout(self.transform(to_conv_layout(x)))


class ChannelNorm(nn.Module):
    """BatchNorm over the channel axis of a ``[B, T, N, C]`` tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return to_graph_layout(self.bn(to_conv_layout(x)))


class ResidualProjection(nn.Module):
    """Identity when channel widths match, otherwise a node- and time-wise linear map."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
