#!/usr/bin/env python
"""Determinism smoke test: load a model, run forward twice, assert identical logits.

    uv run python tools/smoke_test.py [--model 70m] [--prompt "..."]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import models
from lib.utils import device_report, select_device, set_all_seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="70m")
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    args = ap.parse_args()

    set_all_seeds(0)
    device = select_device()
    print(f"[smoke] device={device} {device_report(device)}")

    tok = models.load_tokenizer(args.model)
    model = models.load_model(args.model, device=device)
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        a = model(ids).logits
        b = model(ids).logits

    identical = torch.equal(a, b)
    print(f"[smoke] logits={tuple(a.shape)} bit-identical={identical} "
          f"max_abs_diff={(a - b).abs().max().item()}")
    print("[smoke] " + ("PASS" if identical else "FAIL: forward not deterministic"))
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
