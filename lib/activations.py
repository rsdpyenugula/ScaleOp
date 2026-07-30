"""Hook-based activation capture + caching. [M1]

For each transformer block we grab its output hidden state (the residual stream
just after that block). Over all eval sequences we keep two cheap per-layer
summaries — the mean over tokens and the last-token vector — and for the small
CKA subset we keep every token position (needed for full-token CKA in M2).
Everything is stored as float16 under data/activations/<model>/.
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from .data import CKA_SUBSET, DATA_DIR, load_eval_tokens
from .models import load_model
from .utils import select_device

ACT_DIR = DATA_DIR / "activations"


@torch.no_grad()
def capture_activations(model, tokens, *, batch_size: int = 64, cka_subset: int = CKA_SUBSET) -> dict:
    """Run the model over `tokens` and collect per-block activations.

    Registers a forward hook on every transformer block to grab its output
    hidden state (B, seq_len, d_model), then reduces per batch:
      - mean_pooled: average over the sequence  -> (n_layers, n_seqs, d_model)
      - last_token:  the final position          -> (n_layers, n_seqs, d_model)
      - cka_tokens:  all positions, first `cka_subset` seqs only
                     -> (n_layers, cka_subset, seq_len, d_model)
    All eval sequences are exactly seq_len tokens (no padding), so the mean is
    clean. Everything is moved to CPU float16 to keep the cache compact.
    """
    device = next(model.parameters()).device
    layers = model.gpt_neox.layers  # GPT-NeoX only (Phase 1 is all Pythia); abstract if a 2nd arch is added
    n_layers = len(layers)

    mean_pooled: list[list] = [[] for _ in range(n_layers)]
    last_token: list[list] = [[] for _ in range(n_layers)]
    cka: list[list] = [[] for _ in range(n_layers)]

    captured: dict[int, torch.Tensor] = {}  # layer_idx -> hidden of the current batch

    def make_hook(i: int):
        def hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured[i] = hidden.detach()
        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]
    try:
        seen = 0  # eval sequences processed so far (used to slice the CKA subset)
        for start in tqdm(range(0, len(tokens), batch_size), desc="capture", unit="batch"):
            batch = tokens[start:start + batch_size].to(device)
            captured.clear()
            model(batch)
            assert len(captured) == n_layers, "not every block hook fired"
            bsz = batch.shape[0]
            cka_keep = min(max(cka_subset - seen, 0), bsz)   # rows of this batch in the CKA subset
            for i in range(n_layers):
                h = captured.pop(i)                          # (B, seq_len, d_model); freed after use
                mean_pooled[i].append(h.mean(dim=1, dtype=torch.float32).half().cpu())
                last_token[i].append(h[:, -1, :].half().cpu())
                if cka_keep:
                    cka[i].append(h[:cka_keep].half().cpu())
            seen += bsz
    finally:
        for hd in handles:
            hd.remove()

    def stack_layers(per_layer):
        return torch.stack([torch.cat(batches, dim=0) for batches in per_layer])

    return {
        "mean_pooled": stack_layers(mean_pooled),
        "last_token": stack_layers(last_token),
        "cka_tokens": stack_layers(cka),
        "meta": {"n_layers": n_layers, "n_seqs": len(tokens),
                 "seq_len": tokens.shape[1], "cka_subset": cka_subset},
    }


def capture_and_cache(key: str, *, batch_size: int = 64):
    """Capture activations for one model over the frozen eval set and cache them.

    Loads the model on the best device, runs `capture_activations`, and writes
    mean_pooled / last_token / cka_tokens / meta as separate .pt files under
    data/activations/<key>/. Returns (out_dir, meta).
    """
    tokens = load_eval_tokens()["tokens"]
    model = load_model(key, device=select_device())
    try:
        acts = capture_activations(model, tokens, batch_size=batch_size)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = ACT_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("mean_pooled", "last_token", "cka_tokens"):
        torch.save(acts[name], out_dir / f"{name}.pt")
    torch.save(acts["meta"], out_dir / "meta.pt")
    return out_dir, acts["meta"]


def load_cache(key: str, which: str = "mean_pooled") -> torch.Tensor:
    """Load one cached activation tensor for a model (mean_pooled/last_token/cka_tokens)."""
    return torch.load(ACT_DIR / key / f"{which}.pt", weights_only=True)


@torch.no_grad()
def second_moments(model, tokens, *, batch_size: int = 16) -> dict:
    """Input second moments E[x xᵀ] at the two M5-hybrid compensation points.

    Per layer: the input to mlp.dense_4h_to_h (the post-GELU hidden vector) and
    the input to attention.dense (the attention context). These are what the
    least-squares re-fit needs to make kept wires imitate the full layer.
    Returns {(layer, "mlp_in"): (4h, 4h), (layer, "attn_in"): (d, d)} on device.
    """
    device = next(model.parameters()).device
    mom: dict = {}

    def make(l, name):
        def hook(_m, inp, _out):
            x = inp[0].detach().float().flatten(0, 1)          # (B*T, D)
            mom[(l, name)] = mom.get((l, name), 0) + x.T @ x
        return hook

    hooks = []
    for l, blk in enumerate(model.gpt_neox.layers):
        hooks.append(blk.mlp.dense_4h_to_h.register_forward_hook(make(l, "mlp_in")))
        hooks.append(blk.attention.dense.register_forward_hook(make(l, "attn_in")))
    try:
        for s in tqdm(range(0, len(tokens), batch_size), desc="moments", unit="batch"):
            model(tokens[s:s + batch_size].to(device))
    finally:
        for h in hooks:
            h.remove()
    return mom
