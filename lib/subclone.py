"""Weight subcloning: shrink a model by SELECTING units, not blending them. [M5]

Selection maps are the structure-respecting alternative to dense projection
(M3's diagnosis): keeping whole coordinates preserves rotary pairing, per-head
attention, elementwise GELU and per-feature LayerNorms. Reference recipe:
"Weight subcloning" (Samragh et al., Apple, 2023) — importance-based selection.

Axis rules, primary pair (1.4B → 410M: same 24 layers, same 16 heads — widths shrink):
  - residual stream (2048→1024): ONE global set of feature indices, chosen by
    activation variance on the frozen corpus (a shared stream needs one basis).
  - Q/K head dims (128→64 per head): rotary dims must keep valid (cos,sin)
    pairing AND matching frequencies. GPT-NeoX pairs rotary dim i with
    i + rot/2, freq_i = base^(-2i/rot); with rot 32→16 the small ladder is
    exactly every other large frequency, so we keep pairs {0,2,...,14} — an
    exact frequency match. Non-rotary dims: top-48 of 96 by q/k row norms.
  - V/O head dims: top-64 of 128 per head by ‖V_row‖·‖O_col‖ (the V→O path).
  - MLP hidden (8192→4096): top units by ‖up_row‖·‖down_col‖.

Unseen pair (410M → 160M: 24→12 layers, 16→12 heads, head_dim unchanged) adds:
  - depth: keep evenly-strided blocks (`keep_blocks`, the same 2l mapping as
    M4's transfer test); each kept block's LNs/biases ride along.
  - heads: WHOLE-head selection (head_dim matches so rotary is untouched);
    head importance = Σ_d ‖V_row_d‖·‖O_col_d‖, one keep-set per layer shared
    by Q/K/V/O.
Selections for Q/K (and V/O) are shared so the attention products stay aligned;
biases and LayerNorms ride along with their axis. Output dict plugs straight
into assemble.build_model (ln_source="project" uses the selected LNs — no
target-model leakage anywhere).
"""
from __future__ import annotations

import torch


def residual_selection(large_key: str, d_small: int) -> torch.Tensor:
    """Global residual-feature choice: top-d_small by activation variance,
    averaged over layers, on the frozen corpus. Returns sorted indices."""
    from .activations import load_cache

    acts = load_cache(large_key, "mean_pooled").float()      # (L, N, d_large)
    score = acts.var(dim=1).mean(dim=0)                      # (d_large,)
    return score.topk(d_small).indices.sort().values


def _qk_head_indices(Q, K, heads, head_dim_L, head_dim_S, rotary_pct):
    """Per-head kept dims for Q/K: frequency-matched rotary pairs + top non-rotary."""
    rot_L = int(head_dim_L * rotary_pct)
    rot_S = int(head_dim_S * rotary_pct)
    half_L, half_S = rot_L // 2, rot_S // 2
    stride = half_L // half_S                                # 16/8 = 2: every other freq
    x1 = torch.arange(0, half_L, stride)[:half_S]
    rotary = torch.cat([x1, x1 + half_L])                    # kept (cos, sin) partners
    keep_nonrot = head_dim_S - rot_S
    idx = []
    for h in range(heads):
        rows = slice(h * head_dim_L, (h + 1) * head_dim_L)
        norms = Q[rows].norm(dim=1) + K[rows].norm(dim=1)    # (head_dim_L,)
        nonrot = (norms[rot_L:].topk(keep_nonrot).indices.sort().values + rot_L)
        idx.append(torch.cat([rotary.to(Q.device), nonrot]) + h * head_dim_L)
    return torch.cat(idx)


def _vo_head_indices(V, O, heads, head_dim_L, head_dim_S):
    """Per-head kept dims for V/O: top head_dim_S by ‖V_row‖·‖O_col‖."""
    idx = []
    for h in range(heads):
        rows = slice(h * head_dim_L, (h + 1) * head_dim_L)
        score = V[rows].norm(dim=1) * O[:, rows].norm(dim=0)
        idx.append(score.topk(head_dim_S).indices.sort().values + h * head_dim_L)
    return torch.cat(idx)


def _whole_head_indices(V, O, heads, head_dim_L, heads_small):
    """Keep whole heads (head_dim unchanged): top heads_small by the head's
    total output contribution Σ_d ‖V_row_d‖·‖O_col_d‖."""
    scores = torch.stack([
        (V[h * head_dim_L:(h + 1) * head_dim_L].norm(dim=1)
         * O[:, h * head_dim_L:(h + 1) * head_dim_L].norm(dim=0)).sum()
        for h in range(heads)])
    kept_heads = scores.topk(heads_small).indices.sort().values
    return torch.cat([torch.arange(h * head_dim_L, (h + 1) * head_dim_L, device=V.device)
                      for h in kept_heads.tolist()])


def select_indices(WL: dict, *, heads: int, head_dim_small: int, mlp_small: int,
                   rotary_pct: float, heads_small: int | None = None,
                   keep_blocks: list[int] | None = None) -> dict:
    """Kept-wire indices per SMALL layer: {(l, "qk"|"vo"|"mlp"): indices}.

    keep_blocks maps small layer l -> large block keep_blocks[l] (default: all,
    same depth). If heads shrink (heads_small < heads), head_dim must match and
    whole heads are kept — one keep-set shared by Q/K/V/O per layer.
    """
    heads_small = heads_small or heads
    keep_blocks = keep_blocks or list(range(1 + max(l for l, _ in WL)))
    head_dim_L = WL[(0, "Q")].shape[0] // heads
    if heads_small < heads:
        assert head_dim_small == head_dim_L, "head shrink requires matching head_dim"
    sel: dict = {}
    for l, b in enumerate(keep_blocks):
        Q, K, V, O = WL[(b, "Q")], WL[(b, "K")], WL[(b, "V")], WL[(b, "O")]
        UP, DOWN = WL[(b, "MLP_UP")], WL[(b, "MLP_DOWN")]
        if heads_small < heads:
            idx = _whole_head_indices(V, O, heads, head_dim_L, heads_small)
            sel[(l, "qk")] = sel[(l, "vo")] = idx
        else:
            sel[(l, "qk")] = _qk_head_indices(Q, K, heads, head_dim_L, head_dim_small, rotary_pct)
            sel[(l, "vo")] = _vo_head_indices(V, O, heads, head_dim_L, head_dim_small)
        sel[(l, "mlp")] = (UP.norm(dim=1) * DOWN.norm(dim=0)).topk(mlp_small).indices.sort().values
    return sel


def subclone_weights(WL: dict, res_idx: torch.Tensor, *, heads: int,
                     head_dim_small: int, mlp_small: int, rotary_pct: float,
                     heads_small: int | None = None,
                     keep_blocks: list[int] | None = None,
                     sel: dict | None = None) -> dict:
    """Select a small-shape weight dict out of a large one (extract_weights keys)."""
    res_idx = res_idx.to(next(iter(WL.values())).device)
    keep_blocks = keep_blocks or list(range(1 + max(l for l, _ in WL)))
    sel = sel or select_indices(WL, heads=heads, head_dim_small=head_dim_small,
                                mlp_small=mlp_small, rotary_pct=rotary_pct,
                                heads_small=heads_small, keep_blocks=keep_blocks)
    out: dict = {}
    for l, b in enumerate(keep_blocks):
        Q, K, V, O = WL[(b, "Q")], WL[(b, "K")], WL[(b, "V")], WL[(b, "O")]
        UP, DOWN = WL[(b, "MLP_UP")], WL[(b, "MLP_DOWN")]
        qk, vo, mlp = sel[(l, "qk")], sel[(l, "vo")], sel[(l, "mlp")]

        out[(l, "Q")] = Q[qk][:, res_idx]
        out[(l, "K")] = K[qk][:, res_idx]
        out[(l, "V")] = V[vo][:, res_idx]
        out[(l, "O")] = O[res_idx][:, vo]
        out[(l, "MLP_UP")] = UP[mlp][:, res_idx]
        out[(l, "MLP_DOWN")] = DOWN[res_idx][:, mlp]
        out[(l, "B_Q")] = WL[(b, "B_Q")][qk]
        out[(l, "B_K")] = WL[(b, "B_K")][qk]
        out[(l, "B_V")] = WL[(b, "B_V")][vo]
        out[(l, "B_O")] = WL[(b, "B_O")][res_idx]
        out[(l, "B_MLP_UP")] = WL[(b, "B_MLP_UP")][mlp]
        out[(l, "B_MLP_DOWN")] = WL[(b, "B_MLP_DOWN")][res_idx]
        for t in ("LN1_W", "LN1_B", "LN2_W", "LN2_B"):
            out[(l, t)] = WL[(b, t)][res_idx]
    out[(-1, "EMB_IN")] = WL[(-1, "EMB_IN")][:, res_idx]
    out[(-1, "EMB_OUT")] = WL[(-1, "EMB_OUT")][:, res_idx]
    out[(-1, "LNF_W")] = WL[(-1, "LNF_W")][res_idx]
    out[(-1, "LNF_B")] = WL[(-1, "LNF_B")][res_idx]
    return out


def ls_compensate(M: torch.Tensor, Sigma: torch.Tensor, kept: torch.Tensor,
                  lam: float = 1e-3) -> torch.Tensor:
    """Re-fit a linear map's kept input-columns to imitate the FULL map.

    M (rows, D) reads a D-dim signal with measured second moment Sigma (D, D);
    after cutting the input to `kept` wires, the best (ridge) substitute is
        M' = argmin E‖Mx − M'x_kept‖²  =  M Σ[:,kept] (Σ[kept,kept] + λI)⁻¹
    — the survivors absorb the deleted wires' correlated contribution. Plain
    subcloning is the special case M' = M[:, kept] (no compensation).
    """
    S_fk = Sigma[:, kept]                                   # (D, k)
    S_kk = S_fk[kept]                                       # (k, k)
    reg = lam * S_kk.diagonal().mean()
    eye = torch.eye(len(kept), device=M.device)
    return torch.linalg.solve(S_kk + reg * eye, S_fk.T @ M.T).T


def hybrid_weights(WL: dict, res_idx: torch.Tensor, moments: dict, *, heads: int,
                   head_dim_small: int, mlp_small: int, rotary_pct: float,
                   heads_small: int | None = None,
                   keep_blocks: list[int] | None = None) -> dict:
    """Subclone + least-squares compensation at the two safe (purely linear) spots:
    MLP_DOWN (post-GELU hidden → residual) and O (attention context → residual).
    Read-in weights are NOT compensated in v1: the small LayerNorm renormalizes
    over the kept features, so the compensation target there is ill-defined.
    (Depth reduction drops whole blocks; compensation stays within kept blocks.)
    """
    keep_blocks = keep_blocks or list(range(1 + max(l for l, _ in WL)))
    kw = dict(heads=heads, head_dim_small=head_dim_small, mlp_small=mlp_small,
              rotary_pct=rotary_pct, heads_small=heads_small, keep_blocks=keep_blocks)
    sel = select_indices(WL, **kw)
    W = subclone_weights(WL, res_idx, sel=sel, **kw)
    for l, b in enumerate(keep_blocks):
        W[(l, "MLP_DOWN")] = ls_compensate(WL[(b, "MLP_DOWN")][res_idx],
                                           moments[(b, "mlp_in")], sel[(l, "mlp")])
        W[(l, "O")] = ls_compensate(WL[(b, "O")][res_idx],
                                    moments[(b, "attn_in")], sel[(l, "vo")])
    return W
