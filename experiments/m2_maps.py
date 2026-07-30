#!/usr/bin/env python
"""M2b/c: per-layer width-conversion maps for the primary pair (1.4B -> 410M).

For each matched layer, fit two maps from the large model's pooled activations to
the small model's and score held-out R^2:
  - ridge:      unconstrained linear (2048 -> 1024)
  - procrustes: SVD-reduce to 1024, then rotation + global scale only
The ridge-minus-procrustes gap says how much of the cross-width relationship is a
genuine linear reshaping vs. just a change of basis.

    uv run python experiments/m2_maps.py --config configs/m2.yaml
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
    run = make_run("m2_maps", cfg)
    large, small = cfg["large"], cfg["small"]
    train_frac = cfg.get("train_frac", 0.8)
    lam = cfg.get("ridge_lambda", 1e-3)

    A = load_cache(large, "mean_pooled")   # (L, N, dA)
    B = load_cache(small, "mean_pooled")   # (L, N, dB)
    n_layers, N = A.shape[0], A.shape[1]
    n_train = int(N * train_frac)
    print(f"[m2-maps] {large}{tuple(A.shape)} -> {small}{tuple(B.shape)} | "
          f"train {n_train}/{N} on {run.device}")

    ridge, proc = [], []
    for l in range(n_layers):
        X = A[l].to(run.device, torch.float32)
        Y = B[l].to(run.device, torch.float32)
        ridge.append(alignment.ridge_r2(X, Y, n_train, lam=lam))
        proc.append(alignment.procrustes_r2(X, Y, n_train))

    r_mean = sum(ridge) / n_layers
    p_mean = sum(proc) / n_layers
    run.save_json("maps_r2.json", {
        "large": large, "small": small, "n_train": n_train,
        "ridge_r2": ridge, "procrustes_r2": proc,
        "ridge_r2_mean": r_mean, "procrustes_r2_mean": p_mean,
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(n_layers), ridge, "o-", label=f"ridge (mean {r_mean:.3f})")
    ax.plot(range(n_layers), proc, "s-", label=f"procrustes (mean {p_mean:.3f})")
    ax.set_xlabel("layer"); ax.set_ylabel("held-out R²"); ax.set_ylim(0, 1)
    ax.set_title(f"Width-conversion R² per layer — {large} → {small}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(run.path("maps_r2.png"), dpi=150, bbox_inches="tight")

    print(f"[m2-maps] ridge mean R²={r_mean:.3f}  procrustes mean R²={p_mean:.3f}")
    print(f"[m2-maps] winner: {'ridge' if r_mean > p_mean else 'procrustes'}")
    print(f"[m2-maps] saved -> {run.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
