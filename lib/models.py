"""Pythia model suite: loading and dimension reporting. [M0]

Weight extraction and the fused-QKV split land in M1.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    """The identity + shape of one Pythia model.

    `name` is the HuggingFace repo id used to download it. `layers`, `d_model`
    (hidden width) and `heads` are kept only as a documented *reference* for that
    model's shape — the model's own config is authoritative. `read_dims` reads the
    real dims from the config and warns (never fails) if they differ from these.
    """

    name: str          # HF repo id, e.g. "EleutherAI/pythia-410m"
    layers: int        # number of transformer blocks
    d_model: int       # hidden size (residual-stream width)
    heads: int         # number of attention heads


# The Pythia standard (non-deduped) suite. Pick ONE suite project-wide (Plan §1);
# switch every entry to `-deduped` together if disk allows.
MODEL_SUITE: dict[str, ModelSpec] = {
    "70m":  ModelSpec("EleutherAI/pythia-70m",  layers=6,  d_model=512,  heads=8),
    "160m": ModelSpec("EleutherAI/pythia-160m", layers=12, d_model=768,  heads=12),
    "410m": ModelSpec("EleutherAI/pythia-410m", layers=24, d_model=1024, heads=16),
    "1b":   ModelSpec("EleutherAI/pythia-1b",   layers=16, d_model=2048, heads=8),
    "1.4b": ModelSpec("EleutherAI/pythia-1.4b", layers=24, d_model=2048, heads=16),
    "6.9b": ModelSpec("EleutherAI/pythia-6.9b", layers=32, d_model=4096, heads=32),
}


def resolve_spec(key: str) -> ModelSpec:
    """Look up a model's spec from a short key or its full HF id.

    Lets the rest of the code refer to models by a friendly name: you can pass
    either "410m" or "EleutherAI/pythia-410m" and get back the same ModelSpec.
    Raises KeyError if the name isn't in MODEL_SUITE.
    """
    if key in MODEL_SUITE:
        return MODEL_SUITE[key]
    for spec in MODEL_SUITE.values():
        if spec.name == key:
            return spec
    raise KeyError(f"Unknown model '{key}'. Known: {list(MODEL_SUITE)}")


def load_tokenizer(key: str):
    """Download (or load from cache) the tokenizer for a model.

    The tokenizer turns text into the integer token ids the model reads. Pythia's
    tokenizer has no dedicated padding token, so we point `pad_token` at the
    end-of-sequence token — a standard trick so batched/padded inputs don't error.
    """
    tok = AutoTokenizer.from_pretrained(resolve_spec(key).name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(key: str, *, dtype: torch.dtype = torch.float32, device=None):
    """Download (or load from cache) a Pythia model, ready for inference.

    `dtype` is the numeric precision of the weights — float32 (default) for exact
    analysis math, bfloat16 if you only need forward passes. `device` optionally
    moves the model onto a GPU ("cuda") or Apple GPU ("mps"). The model is put in
    eval mode (no dropout, deterministic) since we never train it here.
    """
    model = AutoModelForCausalLM.from_pretrained(resolve_spec(key).name, dtype=dtype)
    model.eval()
    if device is not None:
        model.to(device)
    return model


def read_dims(model, key: str) -> dict:
    """Read a model's real dims from its own config (the authority).

    The model's config is the source of truth for its shape, so we read layers /
    hidden size / head count straight from it. MODEL_SUITE carries the same
    numbers only as a documented reference; if the two disagree we emit a warning
    (a heads-up that we grabbed an unexpected checkpoint, or that the reference is
    stale) but never raise — a good model should still load and be usable.
    Returns a small summary (real dims + total parameter count) for logs/reports.
    """
    spec = resolve_spec(key)
    dims = {
        "layers": model.config.num_hidden_layers,
        "d_model": model.config.hidden_size,
        "heads": model.config.num_attention_heads,
    }
    reference = {"layers": spec.layers, "d_model": spec.d_model, "heads": spec.heads}
    mismatches = {k: (dims[k], reference[k]) for k in reference if dims[k] != reference[k]}
    if mismatches:
        warnings.warn(
            f"{key}: config dims differ from MODEL_SUITE reference "
            f"(config, reference): {mismatches}",
            stacklevel=2,
        )
    return {"key": key, "hf_id": spec.name, **dims,
            "n_params": sum(p.numel() for p in model.parameters())}


# --- M1: weight extraction -------------------------------------------------
#
# Each extracted tensor is tagged with a (layer_idx, weight_type) key. layer_idx
# is the transformer-block index, or -1 for the model-level tensors (token
# embeddings and the final layernorm). weight_type is one of:
#   Q, K, V     - attention query/key/value projections, split out of the fused
#                 query_key_value matrix (head-aware; see split_fused_qkv)
#   B_Q/B_K/B_V/B_O/B_MLP_UP/B_MLP_DOWN - the matching bias vectors
#   O           - attention output projection            (attention.dense)
#   MLP_UP      - MLP up-projection                       (mlp.dense_h_to_4h)
#   MLP_DOWN    - MLP down-projection                     (mlp.dense_4h_to_h)
#   LN1_W/LN1_B - pre-attention layernorm gain/bias       (input_layernorm)
#   LN2_W/LN2_B - pre-MLP layernorm gain/bias             (post_attention_layernorm)
#   EMB_IN      - token embedding table                   (gpt_neox.embed_in)
#   EMB_OUT     - output/unembedding matrix (lm_head; was embed_out pre-5.x).
#                 Pythia does NOT tie input and output embeddings.
#   LNF_W/LNF_B - final layernorm gain/bias               (final_layer_norm)


def split_fused_qkv(weight, num_heads):
    """Split GPT-NeoX's fused QKV weight into separate Q, K, V matrices.

    GPT-NeoX packs the query/key/value projections into one matrix of shape
    (3*hidden, hidden). The output rows are grouped *by head*, and within each
    head they run [query, key, value] — so a naive top/middle/bottom-third split
    is WRONG. We reshape to (num_heads, 3*head_dim, hidden) and gather each head's
    q/k/v slice, returning three (hidden, hidden) matrices. This head-interleaving
    is the plan's flagged #1-bug risk; `refuse_qkv` lets a test re-fuse and check
    an exact round-trip.
    """
    _, hidden = weight.shape                 # (3*hidden, hidden)
    head_dim = hidden // num_heads
    w = weight.view(num_heads, 3 * head_dim, hidden)
    q = w[:, 0 * head_dim:1 * head_dim, :].reshape(hidden, hidden)
    k = w[:, 1 * head_dim:2 * head_dim, :].reshape(hidden, hidden)
    v = w[:, 2 * head_dim:3 * head_dim, :].reshape(hidden, hidden)
    return q, k, v


def refuse_qkv(q, k, v, num_heads):
    """Re-pack Q, K, V into GPT-NeoX's fused (3*hidden, hidden) layout.

    Exact inverse of `split_fused_qkv`, used by the M1 reconstruction test to
    prove the split loses nothing (re-fused matrix must equal the original).
    """
    hidden = q.shape[0]
    head_dim = hidden // num_heads
    per_head = [t.view(num_heads, head_dim, hidden) for t in (q, k, v)]
    fused = torch.cat(per_head, dim=1)       # (num_heads, 3*head_dim, hidden)
    return fused.reshape(3 * hidden, hidden)


def split_fused_qkv_bias(bias, num_heads):
    """Split the fused QKV bias (3*hidden,) into q/k/v biases, each (hidden,).

    Same head-interleaved layout as the weight: per head the entries run
    [q, k, v]. Inverse is `fuse_qkv_bias`.
    """
    hidden = bias.shape[0] // 3
    head_dim = hidden // num_heads
    b = bias.view(num_heads, 3 * head_dim)
    q = b[:, 0 * head_dim:1 * head_dim].reshape(hidden)
    k = b[:, 1 * head_dim:2 * head_dim].reshape(hidden)
    v = b[:, 2 * head_dim:3 * head_dim].reshape(hidden)
    return q, k, v


def fuse_qkv_bias(q, k, v, num_heads):
    """Re-pack q/k/v biases (hidden,) each into the fused (3*hidden,) layout."""
    per_head = [t.view(num_heads, -1) for t in (q, k, v)]
    return torch.cat(per_head, dim=1).reshape(-1)


def extract_weights(model):
    """Pull out and tag every weight matrix we analyze, from a loaded model.

    Returns a dict of (layer_idx, weight_type) -> tensor. Tensors are detached
    float32 copies, so the model can be freed afterwards. The fused attention QKV
    is split head-aware into separate Q/K/V; head count comes from the model's own
    config (the authority). See the weight_type key above.
    """
    heads = model.config.num_attention_heads
    fp32 = lambda t: t.detach().to(torch.float32).clone()
    out: dict[tuple[int, str], torch.Tensor] = {}
    for i, blk in enumerate(model.gpt_neox.layers):
        q, k, v = split_fused_qkv(blk.attention.query_key_value.weight, heads)
        out[(i, "Q")] = fp32(q)
        out[(i, "K")] = fp32(k)
        out[(i, "V")] = fp32(v)
        out[(i, "O")] = fp32(blk.attention.dense.weight)
        out[(i, "MLP_UP")] = fp32(blk.mlp.dense_h_to_4h.weight)
        out[(i, "MLP_DOWN")] = fp32(blk.mlp.dense_4h_to_h.weight)
        bq, bk, bv = split_fused_qkv_bias(blk.attention.query_key_value.bias, heads)
        out[(i, "B_Q")] = fp32(bq)
        out[(i, "B_K")] = fp32(bk)
        out[(i, "B_V")] = fp32(bv)
        out[(i, "B_O")] = fp32(blk.attention.dense.bias)
        out[(i, "B_MLP_UP")] = fp32(blk.mlp.dense_h_to_4h.bias)
        out[(i, "B_MLP_DOWN")] = fp32(blk.mlp.dense_4h_to_h.bias)
        out[(i, "LN1_W")] = fp32(blk.input_layernorm.weight)
        out[(i, "LN1_B")] = fp32(blk.input_layernorm.bias)
        out[(i, "LN2_W")] = fp32(blk.post_attention_layernorm.weight)
        out[(i, "LN2_B")] = fp32(blk.post_attention_layernorm.bias)
    out[(-1, "EMB_IN")] = fp32(model.gpt_neox.embed_in.weight)
    out[(-1, "EMB_OUT")] = fp32(model.lm_head.weight)
    out[(-1, "LNF_W")] = fp32(model.gpt_neox.final_layer_norm.weight)
    out[(-1, "LNF_B")] = fp32(model.gpt_neox.final_layer_norm.bias)
    return out
