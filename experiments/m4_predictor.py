#!/usr/bin/env python
"""M4 test 3 (decisive): can a predictor R(Ŵ) → Δ beat the shuffle control?

Trains on 18 layers, scores on 6 stratified held-out layers as relative error
reduction vs predicting zero. Control: identical training with Δ shuffled across
layers (within type) — it can still learn per-type mean structure, so only
layer-specific signal separates real from control. 1000-bootstrap CI on the gap.

Needs data/deltas/*.pt from m4_residuals.py (Ŵ recovered as W_small − Δ).

    uv run python experiments/m4_predictor.py --config configs/m2.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.data import DATA_DIR  # noqa: E402
from lib.models import extract_weights, load_model  # noqa: E402
from lib.residuals import (TYPES, err_reduction, eval_patches, patch_dataset,  # noqa: E402
                           train_patch_predictor)
from lib.utils import load_config, make_run  # noqa: E402

TEST_LAYERS = [3, 7, 11, 15, 19, 23]


def load_pair_dataset(small_key, delta_dir, device, seed):
    """(X, Y, layer_idx) real + (Xc, Yc) layer-shuffled control, all types."""
    model = load_model(small_key, device=device)
    WS_all = extract_weights(model)
    del model
    torch.cuda.empty_cache()
    n_layers = 1 + max(l for l, _ in WS_all)

    g = torch.Generator().manual_seed(seed)
    Xs, Ys, Ls, Xc, Yc = [], [], [], [], []
    for ti, t in enumerate(TYPES):
        D = torch.load(delta_dir / f"{t}.pt", weights_only=True).to(device, torch.float32)
        WS = torch.stack([WS_all[(l, t)] for l in range(n_layers)]).to(device)
        What = (WS - D) / (WS - D).std()
        D = D / D.std()
        X, Y, li = patch_dataset(What, D, ti, device)
        Xs.append(X); Ys.append(Y); Ls.append(li)
        perm = torch.randperm(n_layers, generator=g).to(device)
        Xp, Yp, _ = patch_dataset(What, D, ti, device, shuffle_layers=perm)
        Xc.append(Xp); Yc.append(Yp)
    return (torch.cat(Xs), torch.cat(Ys), torch.cat(Ls)), (torch.cat(Xc), torch.cat(Yc))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2.yaml")
    ap.add_argument("--hidden", type=int, nargs="*", default=[0, 512])  # 0 = linear
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = make_run("m4_predictor", cfg)
    dev = run.device

    (X, Y, li), (Xctl, Yctl) = load_pair_dataset(cfg["small"], DATA_DIR / "deltas",
                                                 dev, int(cfg.get("seed", 0)))
    test = torch.isin(li, torch.tensor(TEST_LAYERS, device=dev))

    results = {}
    for hidden in args.hidden:
        name = "linear" if hidden == 0 else f"mlp{hidden}"
        R = train_patch_predictor(X[~test], Y[~test], hidden, seed=0)
        se_p, se_z = eval_patches(R, X[test], Y[test])
        Rc = train_patch_predictor(Xctl[~test], Yctl[~test], hidden, seed=0)
        se_pc, se_zc = eval_patches(Rc, Xctl[test], Yctl[test])
        real, ctl = err_reduction(se_p, se_z), err_reduction(se_pc, se_zc)

        boots, gb = [], torch.Generator().manual_seed(1)
        for _ in range(1000):
            idx = torch.randint(0, len(se_p), (len(se_p),), generator=gb).to(dev)
            boots.append(err_reduction(se_p[idx], se_z[idx]) - err_reduction(se_pc[idx], se_zc[idx]))
        boots = torch.tensor(boots)
        lo, hi = boots.quantile(0.025).item(), boots.quantile(0.975).item()
        p = (boots <= 0).float().mean().item()
        results[name] = {"err_reduction": real, "err_reduction_shuffle": ctl,
                         "gap_ci95": [lo, hi], "p_value": p}
        print(f"[m4-pred] {name:7s} reduction={real:+.4f}  shuffle={ctl:+.4f}  "
              f"gap CI95=[{lo:+.4f},{hi:+.4f}]  p={p:.3f}")

    run.save_json("predictor.json", {"test_layers": TEST_LAYERS, **results})
    print(f"[m4-pred] saved -> {run.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
