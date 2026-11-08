import torch

from models.hydra import Hydra, selective_scan, shift_right


def _naive_scan(x, dt, A, B, C):
    """Direct O(T^2) evaluation of y_t = sum_{s<=t} C_t^T (prod_{r=s+1..t} exp(dt_r A)) dt_s x_s B_s."""
    batch, length, heads, headdim = x.shape
    y = torch.zeros_like(x)
    for b in range(batch):
        for h in range(heads):
            for t in range(length):
                acc = torch.zeros(headdim, B.shape[-1])
                for s in range(t + 1):
                    decay = torch.exp(dt[b, s + 1 : t + 1, h] * A[h]).prod() if s < t else torch.tensor(1.0)
                    acc += decay * dt[b, s, h] * torch.outer(x[b, s, h], B[b, s])
                y[b, t, h] = acc @ C[b, t]
    return y


def test_selective_scan_matches_naive():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 3, 4)
    dt = torch.rand(2, 6, 3) * 0.5
    A = -torch.rand(3) - 0.1
    B = torch.randn(2, 6, 5)
    C = torch.randn(2, 6, 5)
    assert torch.allclose(selective_scan(x, dt, A, B, C), _naive_scan(x, dt, A, B, C), atol=1e-5)


def test_shift_right_zero_pads_first_position():
    y = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    shifted = shift_right(y)
    assert torch.all(shifted[:, 0] == 0)
    assert torch.equal(shifted[:, 1:], y[:, :-1])


def test_hydra_shapes_and_bidirectionality():
    torch.manual_seed(0)
    mixer = Hydra(d_model=32, d_state=8, d_conv=4, expand=2, headdim=16)
    u = torch.randn(3, 7, 32)
    y = mixer(u)
    assert y.shape == u.shape
    # Changing the last position must influence the first one (backward path).
    u2 = u.clone()
    u2[:, -1] += 1.0
    assert not torch.allclose(mixer(u)[:, 0], mixer(u2)[:, 0])
