"""Fast, download-free tests for linear CKA."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import alignment


def test_cka_identity_is_one():
    torch.manual_seed(0)
    X = torch.randn(500, 32)
    assert abs(alignment.linear_cka(X, X) - 1.0) < 1e-4


def test_cka_invariant_to_orthogonal_map():
    """Linear CKA ignores rotations of the feature axes -> ~1 under an orthogonal map."""
    torch.manual_seed(0)
    X = torch.randn(500, 32)
    Q, _ = torch.linalg.qr(torch.randn(32, 32))
    assert alignment.linear_cka(X, X @ Q) > 0.999


def test_cka_independent_is_low():
    torch.manual_seed(0)
    X = torch.randn(500, 32)
    Y = torch.randn(500, 32)
    assert alignment.linear_cka(X, Y) < 0.5


def test_ridge_r2_recovers_linear_map():
    torch.manual_seed(0)
    X = torch.randn(2000, 16)
    Y = X @ torch.randn(16, 8)          # Y is an exact linear map of X
    assert alignment.ridge_r2(X, Y, 1600) > 0.99


def test_ridge_r2_independent_is_low():
    torch.manual_seed(0)
    X, Y = torch.randn(2000, 16), torch.randn(2000, 8)
    assert alignment.ridge_r2(X, Y, 1600) < 0.2


def test_procrustes_r2_recovers_rotation():
    torch.manual_seed(0)
    X = torch.randn(2000, 8)
    Q, _ = torch.linalg.qr(torch.randn(8, 8))
    assert alignment.procrustes_r2(X, X @ Q, 1600) > 0.99
