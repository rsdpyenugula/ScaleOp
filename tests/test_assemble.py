"""Synthetic shape test for project_model — catches basis/axis mix-ups fast.

Uses tiny fake dims (d_large=8 -> d_small=4, 2 heads, 4x MLP, vocab 16) so it
needs no downloads and runs in milliseconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.assemble import project_model
from lib.projection import fit_operator

D_L, D_S, VOCAB, LAYERS = 8, 4, 16, 2


def _fake_large():
    torch.manual_seed(0)
    W = {}
    for l in range(LAYERS):
        for t in ("Q", "K", "V", "O"):
            W[(l, t)] = torch.randn(D_L, D_L)
        W[(l, "MLP_UP")] = torch.randn(4 * D_L, D_L)
        W[(l, "MLP_DOWN")] = torch.randn(D_L, 4 * D_L)
        for t in ("B_Q", "B_K", "B_V", "B_O", "B_MLP_DOWN", "LN1_W", "LN1_B", "LN2_W", "LN2_B"):
            W[(l, t)] = torch.randn(D_L)
        W[(l, "B_MLP_UP")] = torch.randn(4 * D_L)
    W[(-1, "EMB_IN")] = torch.randn(VOCAB, D_L)
    W[(-1, "EMB_OUT")] = torch.randn(VOCAB, D_L)
    W[(-1, "LNF_W")] = torch.randn(D_L)
    W[(-1, "LNF_B")] = torch.randn(D_L)
    return W


def test_project_model_shapes():
    P_res = torch.linalg.qr(torch.randn(D_L, D_S))[0].T   # orthonormal (D_S, D_L)
    out = project_model(_fake_large(), P_res)
    for l in range(LAYERS):
        for t in ("Q", "K", "V", "O"):
            assert out[(l, t)].shape == (D_S, D_S), (l, t)
        assert out[(l, "MLP_UP")].shape == (4 * D_S, D_S)
        assert out[(l, "MLP_DOWN")].shape == (D_S, 4 * D_S)
        for t in ("B_Q", "B_K", "B_V", "B_O", "B_MLP_DOWN", "LN1_W", "LN1_B", "LN2_W", "LN2_B"):
            assert out[(l, t)].shape == (D_S,), (l, t)
        assert out[(l, "B_MLP_UP")].shape == (4 * D_S,)
    assert out[(-1, "EMB_IN")].shape == (VOCAB, D_S)
    assert out[(-1, "EMB_OUT")].shape == (VOCAB, D_S)
    assert out[(-1, "LNF_W")].shape == (D_S,)


def test_fit_operator_handles_rank_limited_mlp_shapes():
    """MLP stacks have out_S > rank(mean weight); init must fall back to QR."""
    torch.manual_seed(0)
    WL = torch.randn(2, 4 * D_L, D_L)     # e.g. MLP_UP: (layers, 32, 8), rank 8
    WS = torch.randn(2, 4 * D_S, D_S)     # target (layers, 16, 4); 16 > 8
    pred, err = fit_operator(WL, WS, steps=5)
    assert pred.shape == WS.shape and err > 0
