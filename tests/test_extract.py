"""M1 weight extraction — the real-model reconstruction check (uses the 70m).

Loads a model (unlike test_models.py's synthetic tests), so it's separated out.
Proves the head-aware QKV split loses nothing on an actual Pythia checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import models


def test_extract_weights_reconstructs_qkv():
    model = models.load_model("70m")
    heads = model.config.num_attention_heads
    hidden = model.config.hidden_size
    W = models.extract_weights(model)

    # Each block's split Q/K/V must re-fuse exactly to the original fused matrix.
    for i, blk in enumerate(model.gpt_neox.layers):
        original = blk.attention.query_key_value.weight.detach().float()
        refused = models.refuse_qkv(W[(i, "Q")], W[(i, "K")], W[(i, "V")], heads)
        assert torch.equal(refused, original), f"block {i} QKV re-fuse mismatch"

    # Spot-check tags and shapes.
    assert W[(0, "Q")].shape == (hidden, hidden)
    assert W[(-1, "EMB_IN")].shape[1] == hidden
    assert (-1, "EMB_OUT") in W and (-1, "LNF_W") in W
