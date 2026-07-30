#!/usr/bin/env python
"""Download the Pythia suite and report each model's config dims + param count.

Dims come from each model's own config (authoritative); read_dims warns if they
differ from the MODEL_SUITE reference but never fails.

    uv run python tools/download_models.py [--models 70m 160m 410m 1b 1.4b]
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(models.MODEL_SUITE))
    args = ap.parse_args()

    for key in args.models:
        print(f"[dl] {key} ({models.resolve_spec(key).name}) ...", flush=True)
        model = models.load_model(key)
        info = models.read_dims(model, key)  # reads config; warns on any mismatch
        print(f"[dl]   layers={info['layers']} d_model={info['d_model']} "
              f"heads={info['heads']} params={info['n_params'] / 1e6:.1f}M")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[dl] done ({len(args.models)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
