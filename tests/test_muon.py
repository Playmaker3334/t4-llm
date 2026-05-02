import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm125m.training.muon import Muon, newton_schulz_orthogonalize


def test_newton_schulz_produces_near_orthogonal_for_square():
    torch.manual_seed(0)
    G = torch.randn(64, 64).double()
    Q = newton_schulz_orthogonalize(G.float(), steps=5).double()
    sv = torch.linalg.svdvals(Q)
    median_dev = (sv - 1.0).abs().median().item()
    assert median_dev < 0.2, f"median singular value deviation too large: {median_dev}"
    pct_near_one = ((sv - 1.0).abs() < 0.35).float().mean().item()
    assert pct_near_one > 0.85, f"only {pct_near_one*100:.1f}% of singular values near 1"


def test_newton_schulz_handles_tall_matrix():
    torch.manual_seed(0)
    G = torch.randn(128, 32)
    Q = newton_schulz_orthogonalize(G, steps=5)
    assert Q.shape == G.shape
    assert torch.isfinite(Q).all()


def test_newton_schulz_handles_wide_matrix():
    torch.manual_seed(0)
    G = torch.randn(32, 128)
    Q = newton_schulz_orthogonalize(G, steps=5)
    assert Q.shape == G.shape
    assert torch.isfinite(Q).all()


def test_muon_step_changes_params():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(64, 128))
    p_before = p.detach().clone()
    p.grad = torch.randn_like(p)
    opt = Muon([p], lr=0.01, momentum=0.9, weight_decay=0.0)
    opt.step()
    diff = (p - p_before).norm().item()
    assert diff > 0


def test_muon_rejects_non_2d():
    p = torch.nn.Parameter(torch.randn(32))
    p.grad = torch.randn_like(p)
    opt = Muon([p], lr=0.01)
    try:
        opt.step()
    except ValueError:
        return
    assert False, "expected ValueError for 1D parameter"


def test_muon_weight_decay_shrinks_params():
    p = torch.nn.Parameter(torch.ones(32, 32))
    p.grad = torch.zeros_like(p)
    opt = Muon([p], lr=0.1, momentum=0.0, weight_decay=0.5)
    p_norm_before = p.norm().item()
    opt.step()
    p_norm_after = p.norm().item()
    assert p_norm_after < p_norm_before
