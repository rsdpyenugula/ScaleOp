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


def each_model(device):
    cfg = AutoConfig.from_pretrained(resolve_spec("410m").name)
    for arm in ("hybrid", "subclone", "projection", "random"):
        model = GPTNeoXForCausalLM(cfg)
        model.load_state_dict(torch.load(CKPT_DIR / f"{arm}.pt", weights_only=True))
        yield arm, model.to(device).eval()
    yield "real_410m", load_model("410m", device=device)


def main() -> int:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    run = make_run("m5_extended_eval", {"seed": 0})
    tok = load_tokenizer("410m")
    results = {}
    for name, model in each_model(run.device):
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
