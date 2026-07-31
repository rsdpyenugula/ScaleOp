#!/usr/bin/env python
"""Upload the race checkpoints to a PRIVATE Hugging Face model repo.

Run on the machine that holds data/m5/. Visibility stays private until the
paper is public. Repo layout mirrors the paper: pair / init / budget / seed.

    uv run python tools/hf_upload.py [--repo <user>/scaleop-pythia-conversions]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

FILES = {  # local path (under data/m5/) -> path in the HF repo
    "30M/hybrid.pt":      "pairA_1.4b-to-410m/hybrid_30M.pt",
    "30M/subclone.pt":    "pairA_1.4b-to-410m/subclone_30M.pt",
    "30M/projection.pt":  "pairA_1.4b-to-410m/projection_30M.pt",
    "30M/random.pt":      "pairA_1.4b-to-410m/random_30M.pt",
    "hybrid.pt":          "pairA_1.4b-to-410m/hybrid_100M.pt",
    "subclone.pt":        "pairA_1.4b-to-410m/subclone_100M.pt",
    **{f"{arm}_b_s{s}.pt": f"pairB_410m-to-160m/{arm}_30M_s{s}.pt"
       for arm in ("hybrid", "subclone", "random") for s in (0, 1, 2)},
    "hybrid_b100.pt":     "pairB_410m-to-160m/hybrid_100M_s0.pt",
    "subclone_b100.pt":   "pairB_410m-to-160m/subclone_100M_s0.pt",
}

CARD = """---
license: apache-2.0
---
# ScaleOp — Pythia size-conversion checkpoints (paper artifacts)

GPT-NeoX (Pythia-architecture) checkpoints from matched-budget recovery races
comparing size-conversion initializations: selection ("subcloning-style"),
dense projection, selection + least-squares compensation (hybrid), and random.
Pairs: 1.4B→410M (width-dominated) and 410M→160M (depth-dominated, 3 seeds).
Files are `state_dict`s for `GPTNeoXForCausalLM` with the target size's config.

Paper + code: see the linked GitHub repository (public with the paper).
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="YenugulaAIML/scaleop-pythia-conversions")
    ap.add_argument("--dir", default="data/m5")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=args.repo)
    base = Path(args.dir)
    for local, remote in FILES.items():
        p = base / local
        if not p.exists():
            print(f"[hf] SKIP (missing): {p}")
            continue
        print(f"[hf] uploading {p} -> {remote} ({p.stat().st_size / 1e9:.2f} GB)", flush=True)
        api.upload_file(path_or_fileobj=str(p), path_in_repo=remote, repo_id=args.repo)
    print(f"[hf] done -> https://huggingface.co/{args.repo} (private)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
