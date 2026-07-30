#!/usr/bin/env python
"""M2a: linear-CKA layer-correspondence heatmap for the primary pair (1.4B vs 410M).

Uses the cached CKA-subset activations (all token positions) from M1. Confirms
that the same-depth pair lines up 1:1 (dominant diagonal) before we fit any
width-conversion map.

    uv run python experiments/m2_cka.py --config configs/m2.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import alignment  # noqa: E402
from lib.activations import load_cache  # noqa: E402
from lib.utils import load_config, make_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = make_run("m2_cka", cfg)
    large, small = cfg["large"], cfg["small"]

    # (L, S, T, d) -> (L, S*T, d): every (sequence, position) is a CKA sample.
    A = load_cache(large, "cka_tokens").flatten(1, 2)
    B = load_cache(small, "cka_tokens").flatten(1, 2)
    print(f"[m2-cka] {large} {tuple(A.shape)} vs {small} {tuple(B.shape)} on {run.device}")

    M = alignment.cka_matrix(A, B, device=str(run.device))

    # Diagonal-dominance summary (both models have the same depth here).
    diag = M.diag()
    argmax_on_diag = (M.argmax(dim=1) == torch.arange(M.shape[0])).float().mean().item()
    run.save_json("cka.json", {
        "large": large, "small": small,
        "matrix": M.tolist(),
        "diag_mean": diag.mean().item(),
        "offdiag_mean": (M.sum() - diag.sum()).item() / (M.numel() - len(diag)),
        "argmax_on_diag_frac": argmax_on_diag,
    })

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M.numpy(), origin="upper", vmin=0, vmax=1, cmap="viridis")
    ax.set_xlabel(f"{small} layer")
    ax.set_ylabel(f"{large} layer")
    ax.set_title(f"Linear CKA — {large} vs {small}")
    fig.colorbar(im, label="CKA")
    fig.savefig(run.path("cka_heatmap.png"), dpi=150, bbox_inches="tight")

    print(f"[m2-cka] diag_mean={diag.mean():.3f}  argmax_on_diag={argmax_on_diag:.2f}")
    print(f"[m2-cka] saved -> {run.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
