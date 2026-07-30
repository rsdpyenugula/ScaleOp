#!/usr/bin/env python
"""Capture & cache block activations for one or more models over the eval set.

    uv run python tools/capture_activations.py [--models 70m 160m ...] [--batch-size 64]

Caches to data/activations/<model>/. Process models one at a time (never hold
several big models + caches in memory together).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import activations, models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(models.MODEL_SUITE))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    for key in args.models:
        t0 = time.time()
        print(f"[capture] {key} ...", flush=True)
        out_dir, meta = activations.capture_and_cache(key, batch_size=args.batch_size)
        secs = time.time() - t0
        mb = sum(p.stat().st_size for p in out_dir.glob("*.pt")) / 1e6
        print(f"[capture]   {meta} -> {out_dir} ({mb:.0f} MB, {secs:.0f}s)")

    print("[capture] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
