#!/usr/bin/env python
"""M3a: target-free projection of 1.4B weights into 410M shape + residual size.

Projects the large model with assemble.project_model (the same weights the
assembled model runs with — one source of truth) and measures the relative
Frobenius residual to the ACTUAL 410M weight per type. That residual is what M4
analyzes. For contrast, fit_operator (which peeks at the target) shows how
linearly the two sizes relate at best.

    uv run python experiments/m3_project.py --config configs/m2.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import assemble, projection  # noqa: E402
from lib.models import extract_weights, load_model  # noqa: E402
from lib.utils import load_config, make_run  # noqa: E402

TYPES = ["Q", "K", "V", "O", "MLP_UP", "MLP_DOWN"]


def _weights(key, device):
    model = load_model(key, device=device)
    try:
        return extract_weights(model)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2.yaml")
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = make_run("m3_project", cfg)
    large, small = cfg["large"], cfg["small"]
    dev = run.device

    P_res = assemble.residual_basis(large, small).to(dev)
    print(f"[m3-proj] P_res {tuple(P_res.shape)} from {large} residual activations")
    WL = {k: v.to(dev) for k, v in _weights(large, dev).items()}
    WS = {k: v.to(dev) for k, v in _weights(small, dev).items()}
    W_hat = assemble.project_model(WL, P_res)

    n_layers = 1 + max(l for l, _ in WS)
    rows = {}
    for t in TYPES:
        errs = [projection.relative_error(WS[(l, t)], W_hat[(l, t)]) for l in range(n_layers)]
        proj_err = sum(errs) / len(errs)
        stack = lambda W: torch.stack([W[(l, t)] for l in range(n_layers)])
        _, op_err = projection.fit_operator(stack(WL), stack(WS), steps=args.steps)
        rows[t] = {"projection": proj_err, "operator": op_err}
        print(f"[m3-proj] {t:9s} projection={proj_err:.3f}  operator={op_err:.3f}")

    run.save_json("projection_error.json", {"large": large, "small": small,
                                            "metric": "relative_frobenius", "types": rows})
    print(f"[m3-proj] saved -> {run.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
