"""M1 activation capture — shape + determinism check (uses the 70m).

Skipped where the frozen eval corpus isn't built (e.g. a fresh machine); runs on
the Spark where data/eval_tokens.pt exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import activations
from lib.data import EVAL_TOKENS_PATH, load_eval_tokens
from lib.models import load_model
from lib.utils import select_device, set_all_seeds


@pytest.mark.skipif(not EVAL_TOKENS_PATH.exists(), reason="eval_tokens.pt not built")
def test_capture_shapes_and_determinism():
    set_all_seeds(0)
    tokens = load_eval_tokens()["tokens"][:128]
    model = load_model("70m", device=select_device())
    a1 = activations.capture_activations(model, tokens, batch_size=64)
    a2 = activations.capture_activations(model, tokens, batch_size=64)

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    seq_len = tokens.shape[1]
    assert a1["mean_pooled"].shape == (n_layers, 128, d_model)
    assert a1["last_token"].shape == (n_layers, 128, d_model)
    assert a1["cka_tokens"].shape == (n_layers, 128, seq_len, d_model)

    assert torch.equal(a1["mean_pooled"], a2["mean_pooled"])
    assert torch.equal(a1["cka_tokens"], a2["cka_tokens"])
