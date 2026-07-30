"""Fast, download-free sanity tests (M0 + the M1 fused-QKV split)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import models
from lib.utils import select_device, set_all_seeds


def test_model_suite_dims_are_consistent():
    for spec in models.MODEL_SUITE.values():
        assert spec.name.startswith("EleutherAI/pythia-")
        assert spec.d_model % spec.heads == 0


def test_resolve_spec_accepts_key_and_hf_id():
    assert models.resolve_spec("410m").name == "EleutherAI/pythia-410m"
    assert models.resolve_spec("EleutherAI/pythia-410m").layers == 24


def test_device_selection_and_seed():
    set_all_seeds(0)
    assert select_device().type in {"cuda", "mps", "cpu"}


def test_qkv_split_refuse_roundtrip():
    """split -> refuse must return the original fused matrix exactly."""
    torch.manual_seed(0)
    heads, hidden = 4, 32
    fused = torch.randn(3 * hidden, hidden)
    q, k, v = models.split_fused_qkv(fused, heads)
    assert q.shape == (hidden, hidden)
    assert torch.equal(models.refuse_qkv(q, k, v, heads), fused)


def test_qkv_bias_split_fuse_roundtrip():
    """Bias split -> fuse must return the original fused bias exactly."""
    torch.manual_seed(0)
    heads, hidden = 4, 32
    fused = torch.randn(3 * hidden)
    q, k, v = models.split_fused_qkv_bias(fused, heads)
    assert torch.equal(models.fuse_qkv_bias(q, k, v, heads), fused)


def test_qkv_split_is_head_interleaved():
    """Guards against a naive third-split: per head the rows run [q, k, v]."""
    heads, hidden = 2, 8
    head_dim = hidden // heads
    one_head = torch.cat([torch.full((head_dim, hidden), 1.0),   # q rows
                          torch.full((head_dim, hidden), 2.0),   # k rows
                          torch.full((head_dim, hidden), 3.0)])  # v rows
    fused = one_head.repeat(heads, 1)          # (3*hidden, hidden), head-grouped
    q, k, v = models.split_fused_qkv(fused, heads)
    assert torch.all(q == 1) and torch.all(k == 2) and torch.all(v == 3)
