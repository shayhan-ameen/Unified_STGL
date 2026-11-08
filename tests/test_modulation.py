import torch

from models.st_tagcn import TASM, TATM, build_modulation, local_temporal_average, transition_scores


def test_transition_scores_range_and_first_row():
    x = torch.relu(torch.randn(2, 6, 5, 8))
    for measure in ("cosine", "l1", "l2"):
        delta = transition_scores(x, measure)
        assert delta.shape == (2, 6, 5)
        assert delta.min() >= 0 and delta.max() <= 1
        assert torch.equal(delta[:, 0], delta[:, 1])


def test_cosine_transition_is_zero_for_identical_frames():
    frame = torch.relu(torch.randn(1, 1, 4, 8))
    x = frame.repeat(1, 5, 1, 1)
    assert torch.allclose(transition_scores(x, "cosine"), torch.zeros(1, 5, 4), atol=1e-6)


def test_local_temporal_average_with_replicate_padding():
    delta = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # [1, T=4, N=1]
    out = local_temporal_average(delta, radius=1)
    expected = torch.tensor([[[4.0 / 3], [2.0], [3.0], [11.0 / 3]]])
    assert torch.allclose(out, expected)


def test_tasm_tatm_residual_modulation():
    x = torch.relu(torch.randn(2, 6, 5, 8))
    delta = transition_scores(x)
    out_sp, alpha_sp = TASM(radius=1)(x, delta)
    out_tm, alpha_tm = TATM()(x, delta)
    assert alpha_sp.shape == (2, 6, 5) and alpha_tm.shape == (2, 6)
    assert torch.allclose(out_sp, x * (1 + alpha_sp.unsqueeze(-1)))
    assert torch.allclose(out_tm, x * (1 + alpha_tm[:, :, None, None]))


def test_every_modulation_variant_runs():
    x = torch.relu(torch.randn(2, 6, 5, 8))
    delta = transition_scores(x)
    for kind in ("transition", "se", "cbam", "st_selfattn", "static_learnable"):
        for spatial in (True, False):
            module = build_modulation(kind, 8, spatial, 1, 6, 5)
            out, alpha = module(x, delta)
            assert out.shape == x.shape
            assert alpha.shape[:2] == (2, 6)
