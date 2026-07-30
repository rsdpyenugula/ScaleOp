#!/usr/bin/env python
"""M4 tests 1+2: spectral structure and cross-layer consistency of Δ vs controls.

Δ(l, type) = W_410M − A·W_1.4B·Bᵀ with the fitted shared operator (M3 option 1).
Per type: mean effective rank of Δ vs shuffled/Gaussian controls, and mean
cross-layer cosine vs shuffle. Δ stacks are saved (fp16) for the predictor test.

    uv run python experiments/m4_residuals.py --config configs/m2.yaml
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

from lib import residuals  # noqa: E402
from lib.data import DATA_DIR  # noqa: E402
from lib.models import extract_weights, load_model  # noqa: E402
from lib.projection import fit_operator  # noqa: E402
from lib.utils import load_config, make_run  # noqa: E402

TYPES = ["Q", "K", "V", "O", "MLP_UP", "MLP_DOWN"]
DELTA_DIR = DATA_DIR / "deltas"


def _stacks(key, device):
    model = load_model(key, device=device)
    try:
        W = extract_weights(model)
    finally:
        del model
        torch.cuda.empty_cache()
    n = 1 + max(l for l, _ in W)
    return {t: torch.stack([W[(l, t)] for l in range(n)]).to(device) for t in TYPES}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2.yaml")
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = make_run("m4_residuals", cfg)
    WL, WS = _stacks(cfg["large"], run.device), _stacks(cfg["small"], run.device)

    DELTA_DIR.mkdir(parents=True, exist_ok=True)
    out, fig_axes = {}, plt.subplots(2, 3, figsize=(14, 8))
    for i, t in enumerate(TYPES):
        What, err = fit_operator(WL[t], WS[t], steps=args.steps)
        D = WS[t] - What
        torch.save(D.half().cpu(), DELTA_DIR / f"{t}.pt")

        er = [residuals.effective_rank(D[l]) for l in range(D.shape[0])]
        er_sh = [residuals.effective_rank(residuals.shuffle_control(D[l], seed=l)) for l in range(D.shape[0])]
        er_ga = [residuals.effective_rank(residuals.gaussian_control(D[l], seed=l)) for l in range(D.shape[0])]
        cos = residuals.cross_layer_cosine(D)
        cos_sh = residuals.cross_layer_cosine(
            torch.stack([residuals.shuffle_control(D[l], seed=100 + l) for l in range(D.shape[0])]))

        mean = lambda x: sum(x) / len(x)
        out[t] = {"operator_rel_err": err,
                  "erank": mean(er), "erank_shuffle": mean(er_sh), "erank_gauss": mean(er_ga),
                  "xlayer_cos": cos, "xlayer_cos_shuffle": cos_sh}
        print(f"[m4-res] {t:9s} erank={mean(er):7.1f} (shuf {mean(er_sh):7.1f}, gauss {mean(er_ga):7.1f})  "
              f"xlayer_cos={cos:+.4f} (shuf {cos_sh:+.4f})")

        ax = fig_axes[1].flat[i]
        s = torch.linalg.svdvals(D[12]).cpu()
        s_sh = torch.linalg.svdvals(residuals.shuffle_control(D[12], seed=12)).cpu()
        ax.semilogy(s / s.sum(), label="Δ (layer 12)")
        ax.semilogy(s_sh / s_sh.sum(), label="shuffled", alpha=0.7)
        ax.set_title(t); ax.legend(fontsize=7)

    fig_axes[0].suptitle("Residual singular-value spectra vs shuffled control")
    fig_axes[0].savefig(run.path("delta_spectra.png"), dpi=150, bbox_inches="tight")
    run.save_json("residual_tests.json", {"large": cfg["large"], "small": cfg["small"], "types": out})
    print(f"[m4-res] deltas cached -> {DELTA_DIR}")
    print(f"[m4-res] saved -> {run.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
