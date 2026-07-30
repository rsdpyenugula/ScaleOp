"""Synthetic shape tests for subclone_weights, both selection regimes.

Regime A (primary pair): same depth/heads, head dims shrink.
Regime B (unseen pair): depth halves + whole-head selection, head_dim fixed.
Tiny fake dims, no downloads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.subclone import subclone_weights
from test_assemble import _fake_large, D_L, LAYERS, VOCAB  # reuse the fake model


def _shapes_ok(W, layers, d, mlp):
    for l in range(layers):
        assert W[(l, "Q")].shape == (d, d)
        assert W[(l, "MLP_UP")].shape == (mlp, d)
        assert W[(l, "MLP_DOWN")].shape == (d, mlp)
        assert W[(l, "LN1_W")].shape == (d,)
    assert W[(-1, "EMB_IN")].shape == (VOCAB, d)
    assert (layers, "Q") not in W                       # no extra layers


def test_subclone_same_depth_dims_shrink():
    """Regime A: head count kept, head_dim 4->2. rotary_pct=1.0 keeps both rotary
    blocks even (4->2 dims) so the frequency-pair logic applies cleanly."""
    res_idx = torch.arange(0, D_L, 2)                   # 8 -> 4 residual dims
    W = subclone_weights(_fake_large(), res_idx, heads=2, head_dim_small=2,
                         mlp_small=2 * D_L, rotary_pct=1.0)
    _shapes_ok(W, LAYERS, 4, 2 * D_L)


def test_subclone_depth_and_head_reduction():
    """Regime B: keep every other block (2->1) and 1 of 2 whole heads (head_dim stays 4)."""
    res_idx = torch.arange(0, D_L, 2)
    W = subclone_weights(_fake_large(), res_idx, heads=2, head_dim_small=4,
                         mlp_small=2 * D_L, rotary_pct=0.5,
                         heads_small=1, keep_blocks=[0])
    _shapes_ok(W, 1, 4, 2 * D_L)