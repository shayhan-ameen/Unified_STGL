"""Hydra: bidirectional state-space sequence mixing through a quasiseparable
matrix mixer (Hwang et al., NeurIPS 2024), implemented in plain PyTorch.

The block follows the Mamba-2 style layout used by Hydra:

    u  -> in_proj -> [z | xBC | dt]
    xBC -> depthwise Conv1d (same padding) -> SiLU -> [x | B | C]
    y  = Shift(SS_fwd(x)) + Reverse(Shift(SS_bwd(Reverse(x)))) + D * x
    y  -> gated RMSNorm (gate = SiLU(z)) -> out_proj

``SS`` is the semiseparable (selective) state-space scan with per-head
discretisation step ``dt``, per-head decay ``A``, and input/output
projections ``B`` / ``C`` shared across heads.  ``Shift`` moves the sequence
one position to the right with zero padding, so the forward term at position
t aggregates strictly earlier positions, the backward term strictly later
positions, and the diagonal term ``D * x`` the current position.

The scan is written as an explicit loop over time; sequences in this work are
short (at most 64 positions), so no chunked kernel is needed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormGated(nn.Module):
    """RMS normalisation followed by multiplicative gating, as in Mamba-2."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x = x * F.silu(gate)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def selective_scan(
    x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
) -> torch.Tensor:
    """Semiseparable state-space scan.

    Parameters
    ----------
    x  : ``[batch, length, heads, headdim]`` inputs.
    dt : ``[batch, length, heads]`` positive step sizes.
    A  : ``[heads]`` negative decay rates.
    B  : ``[batch, length, d_state]`` input projection (shared across heads).
    C  : ``[batch, length, d_state]`` output projection (shared across heads).

    Returns ``y`` with the shape of ``x`` where
    ``h_t = exp(dt_t A) h_{t-1} + dt_t (x_t ⊗ B_t)`` and ``y_t = h_t C_t``.
    """
    batch, length, heads, headdim = x.shape
    d_state = B.shape[-1]
    state = x.new_zeros(batch, heads, headdim, d_state)
    outputs = []
    for t in range(length):
        decay = torch.exp(dt[:, t] * A)                                  # [batch, heads]
        state = state * decay[:, :, None, None] + (
            (dt[:, t, :, None] * x[:, t]).unsqueeze(-1) * B[:, t, None, None, :]
        )
        outputs.append(torch.einsum("bhpn,bn->bhp", state, C[:, t]))
    return torch.stack(outputs, dim=1)


def shift_right(y: torch.Tensor) -> torch.Tensor:
    """Shift along the time axis (dim 1) by one position with zero padding."""
    return F.pad(y[:, :-1], (0, 0, 0, 0, 1, 0)) if y.dim() == 4 else F.pad(y[:, :-1], (0, 0, 1, 0))


class Hydra(nn.Module):
    """Bidirectional quasiseparable sequence mixer.

    Parameters
    ----------
    d_model : input and output width (C_L in the paper).
    d_state : state dimension of the SSM.
    d_conv  : kernel size of the depthwise temporal convolution.
    expand  : inner width multiplier.
    headdim : channels per SSM head.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.headdim = headdim
        if self.d_inner % headdim != 0:
            raise ValueError("expand * d_model must be divisible by headdim")
        self.nheads = self.d_inner // headdim

        conv_dim = self.d_inner + 2 * d_state
        self.in_proj = nn.Linear(d_model, self.d_inner + conv_dim + 2 * self.nheads, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=d_conv, groups=conv_dim, padding=d_conv // 2)

        # Step-size bias initialised so that softplus(dt_bias) lies in [dt_min, dt_max].
        dt = torch.exp(torch.rand(2 * self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        # Decay rates A = -exp(A_log), one per head and direction.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, 2 * self.nheads + 1, dtype=torch.float32)))
        # Per-channel scaling of the current position (diagonal term).
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.norm = RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """``u``: ``[batch, length, d_model]`` -> ``[batch, length, d_model]``."""
        batch, length, _ = u.shape
        z, xBC, dt = torch.split(
            self.in_proj(u), [self.d_inner, self.d_inner + 2 * self.d_state, 2 * self.nheads], dim=-1
        )
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[:, :, :length].transpose(1, 2))
        x, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        x = x.reshape(batch, length, self.nheads, self.headdim)

        dt = F.softplus(dt + self.dt_bias)                                # [batch, length, 2 * heads]
        dt_fwd, dt_bwd = dt.chunk(2, dim=-1)
        A = -torch.exp(self.A_log)
        A_fwd, A_bwd = A.chunk(2)

        y_fwd = selective_scan(x, dt_fwd, A_fwd, B, C)
        y_bwd = selective_scan(x.flip(1), dt_bwd, A_bwd, B.flip(1), C.flip(1))
        y = shift_right(y_fwd) + shift_right(y_bwd).flip(1)
        y = y.reshape(batch, length, self.d_inner) + x.reshape(batch, length, self.d_inner) * self.D

        y = self.norm(y, z)
        return self.out_proj(y)
