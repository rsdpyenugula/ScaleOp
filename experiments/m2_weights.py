#!/usr/bin/env python
"""M2.2: weight-space alignment on the token embeddings (1.4B vs 410M).

The embeddings are the one weight matrix where the two sizes share an axis (the
vocabulary) and differ only in width — so we can align them with a single map,
exactly like the activations. This shows, in WEIGHT space, how much apparent
cross-width difference alignment removes:
  - BEFORE: linear CKA between the raw large/small embedding matrices
  - AFTER : held-out R^2 of a learned ridge / Procrustes width map
(Per-layer weight projection needs input+output maps and is done in M3.)

    uv run python experiments/m2_weights.py --config configs/m2.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import alignment  # noqa: E402
from lib.models import load_model  # noqa: E402
from lib.utils import load_config, make_run  # noqa: E402


def _embeddings(key, device):
    """Return (embed_in, lm_head) weight matrices (vocab, d_model) on `device`."""
    m = load_model(key, device=device)
    try:
        return (m.gpt_neox.embed_in.weight.detach().float(),
                m.lm_head.weight.detach().float())
    finally:
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = make_run("m2_weights", cfg)
    large, small = cfg["large"], cfg["small"]

    in_L, out_L = _embeddings(large, run.device)
    in_S, out_S = _embeddings(small, run.device)

    results = {}
    for name, WL, WS in [("EMB_IN", in_L, in_S), ("EMB_OUT", out_L, out_S)]:
        n_train = int(WL.shape[0] * cfg.get("train_frac", 0.8))  # split the vocab rows
        results[name] = {
            "cka_before": alignment.linear_cka(WL, WS),
            "ridge_r2": alignment.ridge_r2(WL, WS, n_train, lam=cfg.get("ridge_lambda", 1e-3)),
            "procrustes_r2": alignment.procrustes_r2(WL, WS, n_train),
            "shape_large": list(WL.shape), "shape_small": list(WS.shape),
        }
        r = results[name]
        print(f"[m2-wt] {name}: CKA(before)={r['cka_before']:.3f}  "
              f"ridge R²={r['ridge_r2']:.3f}  procrustes R²={r['procrustes_r2']:.3f}")

    run.save_json("weights.json", {"large": large, "small": small, **results})
    print(f"[m2-wt] saved -> {run.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
