#!/usr/bin/env python
"""M4 test 4: does the residual predictor TRANSFER to a held-out model pair?

Pair A = 1.4B→410M (deltas cached by m4_residuals). Pair B = 410M→160M: fit its
own shared operator per type on matched layers (160M layer l ↔ 410M layer 2l,
24→12 depth), then Δ_B = W_160M − Ŵ_B. Train R on ALL pair-A layers, evaluate
error reduction on pair-B. Control: R trained on layer-shuffled pair-A. Patch
features are size-agnostic, so the same R applies across pairs. Any positive
real-minus-control gap on pair B is family-level structure.

    uv run python experiments/m4_transfer.py --config configs/m2.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.data import DATA_DIR  # noqa: E402
from lib.models import extract_weights, load_model  # noqa: E402
from lib.projection import fit_operator  # noqa: E402
from lib.residuals import (TYPES, err_reduction, eval_patches, patch_dataset,  # noqa: E402
                           train_patch_predictor)
from lib.utils import load_config, make_run  # noqa: E402

from m4_predictor import load_pair_dataset  # noqa: E402  (same dir)


def build_pair_b(large_key, small_key, device, steps):
    """Fit operators for the held-out pair and return its (X, Y) dataset."""
    def stacks(key):
        model = load_model(key, device=device)
        try:
            W = extract_weights(model)
        finally:
            del model
            torch.cuda.empty_cache()
        return W

    WL_all, WS_all = stacks(large_key), stacks(small_key)
    nS = 1 + max(l for l, _ in WS_all)
    Xs, Ys = [], []
    for ti, t in enumerate(TYPES):
        WL = torch.stack([WL_all[(2 * l, t)] for l in range(nS)]).to(device)
        WS = torch.stack([WS_all[(l, t)] for l in range(nS)]).to(device)
        What, err = fit_operator(WL, WS, steps=steps)
        D = WS - What
        print(f"[m4-xfer] pair-B {t:9s} operator_rel_err={err:.3f}")
        X, Y, _ = patch_dataset(What / What.std(), D / D.std(), ti, device)
        Xs.append(X); Ys.append(Y)
    return torch.cat(Xs), torch.cat(Ys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2.yaml")
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = make_run("m4_transfer", cfg)
    dev = run.device

    (XA, YA, _), (XAc, YAc) = load_pair_dataset(cfg["small"], DATA_DIR / "deltas",
                                                dev, int(cfg.get("seed", 0)))
    XB, YB = build_pair_b("410m", "160m", dev, args.steps)

    results = {}
    for hidden in (0, 512):
        name = "linear" if hidden == 0 else f"mlp{hidden}"
        se_p, se_z = eval_patches(train_patch_predictor(XA, YA, hidden, seed=0), XB, YB)
        se_pc, _ = eval_patches(train_patch_predictor(XAc, YAc, hidden, seed=0), XB, YB)
        real, ctl = err_reduction(se_p, se_z), err_reduction(se_pc, se_z)

        boots, gb = [], torch.Generator().manual_seed(1)
        for _ in range(1000):
            idx = torch.randint(0, len(se_p), (len(se_p),), generator=gb).to(dev)
            boots.append(err_reduction(se_p[idx], se_z[idx]) - err_reduction(se_pc[idx], se_z[idx]))
        boots = torch.tensor(boots)
        lo, hi = boots.quantile(0.025).item(), boots.quantile(0.975).item()
        p = (boots <= 0).float().mean().item()
        results[name] = {"transfer_reduction": real, "transfer_reduction_shuffle": ctl,
                         "gap_ci95": [lo, hi], "p_value": p}
        print(f"[m4-xfer] {name:7s} B-reduction={real:+.4f}  shuffle={ctl:+.4f}  "
              f"gap CI95=[{lo:+.4f},{hi:+.4f}]  p={p:.3f}")

    run.save_json("transfer.json", {"pair_A": [cfg["large"], cfg["small"]],
                                    "pair_B": ["410m", "160m"], **results})
    print(f"[m4-xfer] saved -> {run.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
