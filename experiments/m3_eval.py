#!/usr/bin/env python
"""M3b: assemble the projected model and benchmark it against the anchors.

Wikitext-103 perplexity for:
  - real 410M (upper anchor)     - real 1.4B (reference)
  - random-init 410M (lower anchor)
  - projected 1.4B->410M with ln_source="small" and "project" (both variants)
Sanity bar (plan): projected must beat random init. Also prints a short greedy
generation from the projected model to show it runs deterministically.

    uv run python experiments/m3_eval.py --config configs/m2.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoConfig, GPTNeoXForCausalLM  # noqa: E402

from lib import assemble  # noqa: E402
from lib.eval import wikitext_perplexity  # noqa: E402
from lib.models import extract_weights, load_model, load_tokenizer, resolve_spec  # noqa: E402
from lib.utils import load_config, make_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = make_run("m3_eval", cfg)
    large, small = cfg["large"], cfg["small"]
    dev = run.device
    tok = load_tokenizer(small)

    print(f"[m3-eval] building projected weights {large} -> {small}")
    P_res = assemble.residual_basis(large, small).to(dev)
    model_L = load_model(large, device=dev)
    WL = extract_weights(model_L)
    del model_L
    torch.cuda.empty_cache()
    W = assemble.project_model({k: v.to(dev) for k, v in WL.items()}, P_res)
    del WL

    def ppl(model) -> float:
        model.to(dev)
        p = wikitext_perplexity(model, tok)
        del model
        torch.cuda.empty_cache()
        return p

    results = {}
    proj = assemble.build_model(W, small, ln_source="small", device=dev)
    prompt = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
    gen = proj.generate(prompt, max_new_tokens=12, do_sample=False)
    print(f"[m3-eval] projected(greedy): {tok.decode(gen[0])!r}")
    results["projected_ln_small"] = ppl(proj)
    results["projected_ln_project"] = ppl(assemble.build_model(W, small, ln_source="project", device=dev))
    results["random_init"] = ppl(GPTNeoXForCausalLM(AutoConfig.from_pretrained(resolve_spec(small).name)))
    results["real_small"] = ppl(load_model(small))
    results["real_large"] = ppl(load_model(large))

    for k, v in results.items():
        print(f"[m3-eval] {k:22s} ppl={v:,.1f}")
    run.save_json("eval.json", {"large": large, "small": small, "wikitext103_ppl": results})
    print(f"[m3-eval] saved -> {run.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
