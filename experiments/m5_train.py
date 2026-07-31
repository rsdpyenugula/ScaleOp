#!/usr/bin/env python
"""M5: fine-tune one init arm with the shared budget; log the recovery curve.

Arms (all obtainable WITHOUT the trained 410M — no target leakage):
  - subclone   : importance-based selection init (lib/subclone.py)
  - projection : M3's target-free dense projection; LayerNorms reset to fresh
                 (gain 1, bias 0) since projected LNs are ill-defined
  - random     : fresh 410M config init (the from-scratch reference)
  - hybrid     : subclone + least-squares compensation of MLP_DOWN and O
                 (second moments measured on 1000 frozen-corpus sequences)
Deviation from the plan's arm list: "projection + R" is dropped — M4 showed R
carries zero signal, and R was fit against the real 410M (target-circular).

    uv run python experiments/m5_train.py --config configs/m5.yaml --init subclone
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoConfig, GPTNeoXForCausalLM  # noqa: E402

from lib import assemble, subclone  # noqa: E402
from lib.data import pile_train_batches  # noqa: E402
from lib.eval import wikitext_perplexity  # noqa: E402
from lib.models import extract_weights, load_model, load_tokenizer, resolve_spec  # noqa: E402
from lib.utils import load_config, make_run  # noqa: E402


def build_init(arm, cfg, device):
    small = cfg["small"]
    if arm == "random":
        return GPTNeoXForCausalLM(AutoConfig.from_pretrained(resolve_spec(small).name)).to(device)

    model_L = load_model(cfg["large"], device=device)
    # transformers 5.x moved rotary_pct into rope_parameters["partial_rotary_factor"]
    rotary_pct = model_L.config.rope_parameters["partial_rotary_factor"]
    WL = extract_weights(model_L)
    moments = None
    if arm in ("hybrid", "hybrid_rs"):
        from lib.activations import second_moments
        from lib.data import load_eval_tokens
        moments = second_moments(model_L, load_eval_tokens()["tokens"][:1000])
    del model_L
    torch.cuda.empty_cache()
    WL = {k: v.to(device) for k, v in WL.items()}

    spec_L, spec = resolve_spec(cfg["large"]), resolve_spec(small)
    if arm in ("subclone", "subclone_rs", "subclone_iso", "hybrid", "hybrid_rs"):
        res_idx = subclone.residual_selection(cfg["large"], spec.d_model)
        stride = spec_L.layers // spec.layers
        kw = dict(heads=spec_L.heads, head_dim_small=spec.d_model // spec.heads,
                  mlp_small=4 * spec.d_model, rotary_pct=rotary_pct,
                  heads_small=spec.heads if spec.heads != spec_L.heads else None,
                  keep_blocks=list(range(0, spec_L.layers, stride))[:spec.layers] if stride > 1 else None)
        if arm.startswith("subclone"):
            W = subclone.subclone_weights(WL, res_idx, rescale={"subclone_rs": "full", "subclone_iso": "smart"}.get(arm, False), **kw)
        else:
            W = subclone.hybrid_weights(WL, res_idx, moments, **kw)
            del moments
            if arm == "hybrid_rs":  # reference-recipe scale on the LN-fronted reads
                r = (WL[(0, "Q")].shape[1] / spec.d_model) ** 0.5
                for l in range(spec.layers):
                    for t in ("Q", "K", "V", "MLP_UP"):
                        W[(l, t)] = W[(l, t)] * r
                W[(-1, "EMB_OUT")] = W[(-1, "EMB_OUT")] * r
    elif arm == "projection":
        if spec_L.layers != spec.layers:
            raise NotImplementedError("projection arm has no depth mapping (pair-B races run without it)")
        P_res = assemble.residual_basis(cfg["large"], small).to(device)
        W = assemble.project_model(WL, P_res)
        for l in range(spec.layers):                       # fresh LNs (projected are ill-defined)
            for t in ("LN1_W", "LN2_W"):
                W[(l, t)] = torch.ones(spec.d_model, device=device)
            for t in ("LN1_B", "LN2_B"):
                W[(l, t)] = torch.zeros(spec.d_model, device=device)
        W[(-1, "LNF_W")] = torch.ones(spec.d_model, device=device)
        W[(-1, "LNF_B")] = torch.zeros(spec.d_model, device=device)
    else:
        raise ValueError(arm)
    return assemble.build_model(W, small, ln_source="project", device=device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m5.yaml")
    ap.add_argument("--init", required=True,
                    choices=["subclone", "subclone_rs", "subclone_iso", "projection", "random", "hybrid", "hybrid_rs"])
    ap.add_argument("--seed", type=int, default=None, help="override config seed; also offsets the data draw")
    ap.add_argument("--tag", default="", help="suffix for run name + checkpoint (multi-pair/seed runs)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    # Seeds vary the data draw too (the pipeline is otherwise deterministic):
    # each seed trains on a disjoint stream offset; arms within a seed share it.
    cfg["skip_docs"] = cfg["skip_docs"] + cfg["seed"] * 25_000
    run = make_run(f"m5_{args.init}{args.tag}", cfg)
    dev = run.device
    tok = load_tokenizer(cfg["small"])

    model = build_init(args.init, cfg, dev)
    model.train()
    quick = lambda: wikitext_perplexity(model.eval(), tok, device=dev,
                                        max_tokens=cfg["quick_eval_tokens"])

    curve = [{"tokens": 0, "ppl": quick()}]
    print(f"[m5-{args.init}] t=0 ppl={curve[0]['ppl']:,.1f}", flush=True)
    model.train()

    steps_total = math.ceil(cfg["tokens"] / (cfg["seq_len"] * cfg["batch_size"]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95),
                            weight_decay=cfg["weight_decay"])
    warmup, t0, seen, next_log, loss_sum, loss_n = cfg["warmup_steps"], time.time(), 0, cfg["log_every_tokens"], 0.0, 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) *
        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total)))))

    batches = pile_train_batches(tok, seq_len=cfg["seq_len"], batch_size=cfg["batch_size"],
                                 skip_docs=cfg["skip_docs"])
    for step, batch in enumerate(batches):
        batch = batch.to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            loss = model(batch, labels=batch).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()
        seen += batch.numel()
        loss_sum += loss.item()
        loss_n += 1

        if seen >= next_log or seen >= cfg["tokens"]:
            ppl = quick()
            model.train()
            curve.append({"tokens": seen, "ppl": ppl, "train_loss": loss_sum / loss_n})
            run.save_json("curve.json", {"init": args.init, "curve": curve})   # crash-safe
            tps = seen / (time.time() - t0)
            print(f"[m5-{args.init}] t={seen / 1e6:.0f}M ppl={ppl:,.1f} "
                  f"loss={loss_sum / loss_n:.3f} ({tps / 1e3:.1f}k tok/s)", flush=True)
            loss_sum, loss_n, next_log = 0.0, 0, next_log + cfg["log_every_tokens"]
        if seen >= cfg["tokens"]:
            break

    final = wikitext_perplexity(model.eval(), tok, device=dev)
    curve.append({"tokens": seen, "ppl_full": final})
    run.save_json("curve.json", {"init": args.init, "curve": curve})
    ckpt = Path("data/m5") / f"{args.init}{args.tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"[m5-{args.init}] FINAL full ppl={final:,.1f} | ckpt -> {ckpt} | {run.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
