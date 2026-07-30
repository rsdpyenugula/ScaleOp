#!/usr/bin/env python
"""M3 mechanical check: rebuild a model from its own extracted weights, unchanged.

extract_weights -> build_model with NO projection must reproduce the original
model's logits exactly. Passing proves the split/fuse, bias, embedding and LN
plumbing is correct — so any quality loss in the projected model comes from the
projection math itself, not assembly bugs.

    uv run python experiments/m3_identity.py [--model 1.4b]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.assemble import build_model  # noqa: E402
from lib.data import load_eval_tokens  # noqa: E402
from lib.models import extract_weights, load_model  # noqa: E402
from lib.utils import select_device, set_all_seeds  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.4b")
    args = ap.parse_args()
    set_all_seeds(0)
    dev = select_device()
    tokens = load_eval_tokens()["tokens"][:8].to(dev)

    real = load_model(args.model, device=dev)
    with torch.no_grad():
        ref = real(tokens).logits
    W = extract_weights(real)
    del real
    torch.cuda.empty_cache()

    rebuilt = build_model(W, args.model, ln_source="project", device=dev)
    with torch.no_grad():
        got = rebuilt(tokens).logits

    diff = (ref - got).abs().max().item()
    print(f"[m3-id] {args.model}: max |logit diff| = {diff:g}")
    print("[m3-id] " + ("PASS (assembly is mechanically exact)" if diff < 1e-3
                        else "FAIL: assembly has a mechanical bug"))
    return 0 if diff < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
