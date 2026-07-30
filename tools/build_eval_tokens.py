#!/usr/bin/env python
"""Build & freeze the eval token set (data/eval_tokens.pt).

    uv run python tools/build_eval_tokens.py [--force]

--force rebuilds even if the frozen file already exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = data.build_eval_tokens(force=args.force)
    t, m = out["tokens"], out["meta"]
    print(f"[eval-tokens] {tuple(t.shape)} tokens from {m['provenance']}")
    print(f"[eval-tokens] tokenizer={m['tokenizer']} cka_subset={m['cka_subset']} created={m['created']}")
    print(f"[eval-tokens] saved -> {data.EVAL_TOKENS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
