"""Project a large model's weights into a small architecture and make it run. [M3]

Two steps, one source of truth:
  1. `project_model` — target-free projection of every 1.4B weight into 410M
     shape. The residual-stream axis uses one global PCA basis (P_res) from the
     large model's activations; internal axes use per-layer SVD bases that are
     SHARED between weights that interact, so their products survive projection:
       - Q and K share one head-space basis     (attention scores q·k)
       - V and O share one head-space basis     (the O @ V path)
       - MLP_UP and MLP_DOWN share the hidden-axis basis
     Biases ride along with their weight's output basis (y = Wx + b -> Pb).
  2. `build_model` — load those weights into a real small-config GPT-NeoX so it
     generates text. LayerNorms are per-feature vectors, so "projecting" them is
     ill-defined; default copies the small model's LNs (ln_source="small"), the
     "project" variant maps the large model's through P_res for comparison.

Known caveat (report, don't hide): rotary position encoding acts on fixed
per-head dimension pairs; any basis change in head space distorts it. The
acceptance bar for M3 is therefore modest — beat random init on perplexity.
"""
from __future__ import annotations

import torch

from .models import MODEL_SUITE, fuse_qkv_bias, load_model, refuse_qkv
from .projection import pca_basis, project_weight, svd_axis_basis


def project_model(WL: dict, P_res: torch.Tensor) -> dict:
    """Project a full large-model weight dict (from extract_weights) to small shape.

    WL: {(layer, type): tensor} in large shapes; P_res: (d_small, d_large) global
    residual basis. Returns the same keys in small shapes. Per layer, the shared
    internal bases are: B_qk from [Q, K] stacked along columns, B_vo from O's
    input columns, B_mlp from [UP, DOWNᵀ] stacked along columns (UP alone has
    rank d_large < the 4*d_small target, so the hidden-axis basis needs both).
    Shared *output* bases always come from column-stacks: the left singular
    vectors then live in the original output space.
    """
    d_small = P_res.shape[0]
    n_layers = 1 + max(l for l, _ in WL)
    out: dict = {}
    for l in range(n_layers):
        Q, K, V, O = WL[(l, "Q")], WL[(l, "K")], WL[(l, "V")], WL[(l, "O")]
        UP, DOWN = WL[(l, "MLP_UP")], WL[(l, "MLP_DOWN")]
        B_qk = svd_axis_basis(torch.cat([Q, K], dim=1), d_small, axis=0)  # head space of q/k
        B_vo = svd_axis_basis(O, d_small, axis=1)                     # head space of v/o
        B_mlp = svd_axis_basis(torch.cat([UP, DOWN.T], dim=1), 4 * d_small, axis=0)  # mlp hidden axis

        out[(l, "Q")] = project_weight(Q, B_qk, P_res)
        out[(l, "K")] = project_weight(K, B_qk, P_res)
        out[(l, "V")] = project_weight(V, B_vo, P_res)
        out[(l, "O")] = project_weight(O, P_res, B_vo)
        out[(l, "MLP_UP")] = project_weight(UP, B_mlp, P_res)
        out[(l, "MLP_DOWN")] = project_weight(DOWN, P_res, B_mlp)
        # biases follow their weight's output basis
        out[(l, "B_Q")] = B_qk @ WL[(l, "B_Q")]
        out[(l, "B_K")] = B_qk @ WL[(l, "B_K")]
        out[(l, "B_V")] = B_vo @ WL[(l, "B_V")]
        out[(l, "B_O")] = P_res @ WL[(l, "B_O")]
        out[(l, "B_MLP_UP")] = B_mlp @ WL[(l, "B_MLP_UP")]
        out[(l, "B_MLP_DOWN")] = P_res @ WL[(l, "B_MLP_DOWN")]
        for t in ("LN1_W", "LN1_B", "LN2_W", "LN2_B"):                # per-feature vecs
            out[(l, t)] = P_res @ WL[(l, t)]
    # embeddings: the vocab axis is shared, only the width reduces
    out[(-1, "EMB_IN")] = WL[(-1, "EMB_IN")] @ P_res.T
    out[(-1, "EMB_OUT")] = WL[(-1, "EMB_OUT")] @ P_res.T
    out[(-1, "LNF_W")] = P_res @ WL[(-1, "LNF_W")]
    out[(-1, "LNF_B")] = P_res @ WL[(-1, "LNF_B")]
    return out


def residual_basis(large_key: str, small_key: str) -> torch.Tensor:
    """The global residual-stream reduction P_res (d_small, d_large), from the
    large model's cached pooled activations across all layers."""
    from .activations import load_cache

    acts = load_cache(large_key, "mean_pooled").flatten(0, 1).float()  # (L*N, d_large)
    return pca_basis(acts, MODEL_SUITE[small_key].d_model)


def build_model(W: dict, small_key: str, *, ln_source: str = "small", device=None):
    """Turn a small-shape weight dict into a runnable GPT-NeoX model.

    Starts from the REAL small model (so config, rotary tables and — when
    ln_source="small" — its LayerNorms are correct-by-construction), then
    overwrites every projected tensor: fused QKV (via refuse_qkv), O, MLP, biases,
    embeddings. ln_source="project" additionally overwrites all LN gains/biases
    with the projected ones. Returns the model in eval mode.
    """
    model = load_model(small_key, device=device)
    heads = model.config.num_attention_heads
    with torch.no_grad():
        for l, blk in enumerate(model.gpt_neox.layers):
            qkv = refuse_qkv(W[(l, "Q")], W[(l, "K")], W[(l, "V")], heads)
            bias = fuse_qkv_bias(W[(l, "B_Q")], W[(l, "B_K")], W[(l, "B_V")], heads)
            blk.attention.query_key_value.weight.copy_(qkv)
            blk.attention.query_key_value.bias.copy_(bias)
            blk.attention.dense.weight.copy_(W[(l, "O")])
            blk.attention.dense.bias.copy_(W[(l, "B_O")])
            blk.mlp.dense_h_to_4h.weight.copy_(W[(l, "MLP_UP")])
            blk.mlp.dense_h_to_4h.bias.copy_(W[(l, "B_MLP_UP")])
            blk.mlp.dense_4h_to_h.weight.copy_(W[(l, "MLP_DOWN")])
            blk.mlp.dense_4h_to_h.bias.copy_(W[(l, "B_MLP_DOWN")])
            if ln_source == "project":
                blk.input_layernorm.weight.copy_(W[(l, "LN1_W")])
                blk.input_layernorm.bias.copy_(W[(l, "LN1_B")])
                blk.post_attention_layernorm.weight.copy_(W[(l, "LN2_W")])
                blk.post_attention_layernorm.bias.copy_(W[(l, "LN2_B")])
        # Pythia pads vocab differently per size (6.9B: 50432, smaller: 50304);
        # the real tokens (first 50254 rows) are identical — slice to target.
        vocab = model.gpt_neox.embed_in.weight.shape[0]
        assert W[(-1, "EMB_IN")].shape[0] >= vocab, "donor vocab smaller than target"
        model.gpt_neox.embed_in.weight.copy_(W[(-1, "EMB_IN")][:vocab])
        model.lm_head.weight.copy_(W[(-1, "EMB_OUT")][:vocab])
        if ln_source == "project":
            model.gpt_neox.final_layer_norm.weight.copy_(W[(-1, "LNF_W")])
            model.gpt_neox.final_layer_norm.bias.copy_(W[(-1, "LNF_B")])
    return model.eval()
