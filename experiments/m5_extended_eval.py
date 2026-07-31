#!/usr/bin/env python
"""M5 hardening (R3): beyond wikitext-1024 ppl for the four race checkpoints.

Per model (4 checkpoints from data/m5/30M/ + the real 410M reference):
  - lm-eval tasks, zero-shot: lambada_openai, arc_easy, hellaswag, piqa
  - C4 validation ppl (out-of-domain: neither the Pile stream nor wikitext)
  - wikitext ppl at 2048 context (2x the fine-tuning length; stresses rotary)

    uv run python experiments/m5_extended_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoConfig, GPTNeoXForCausalLM  # noqa: E402

from lib.eval import wikitext_perplexity  # noqa: E402
from lib.models import load_model, load_tokenizer, resolve_spec  # noqa: E402
from lib.utils import make_run  # noqa: E402

TASKS = ["lambada_openai", "arc_easy", "hellaswag", "piqa"]
CKPT_DIR = Path("data/m5/30M")


def each_model(device, small, ckpt_dir, tag, arms):
    cfg = AutoConfig.from_pretrained(resolve_spec(small).name)
    for arm in arms:
        model = GPTNeoXForCausalLM(cfg)
        model.load_state_dict(torch.load(ckpt_dir / f"{arm}{tag}.pt", weights_only=True))
        yield arm, model.to(device).eval()
    yield f"real_{small}", load_model(small, device=device)


def main() -> int:
    import argparse

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="410m")
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--tag", default="", help="checkpoint suffix, e.g. _b_s0")
    ap.add_argument("--arms", nargs="*", default=["hybrid", "subclone", "projection", "random"])
    args = ap.parse_args()

    run = make_run(f"m5_extended_eval{args.tag}", {"seed": 0})
    tok = load_tokenizer(args.small)
    results = {}
    for name, model in each_model(run.device, args.small, Path(args.ckpt_dir),
                                  args.tag, args.arms):
        row = {
            "c4_ppl": wikitext_perplexity(model, tok, dataset="c4", max_tokens=200_000),
            "wikitext_ppl_ctx2048": wikitext_perplexity(model, tok, seq_len=2048,
                                                        stride=1024, max_tokens=200_000),
        }
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=32)
        out = lm_eval.simple_evaluate(model=lm, tasks=TASKS)
        for t in TASKS:
            row[t] = out["results"][t].get("acc,none")
        results[name] = row
        print(f"[m5-ext] {name:10s} " + "  ".join(f"{k}={v:.4g}" for k, v in row.items()), flush=True)
        del model, lm
        torch.cuda.empty_cache()
        run.save_json("extended_eval.json", results)   # crash-safe

    print(f"[m5-ext] saved -> {run.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
