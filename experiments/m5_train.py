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
import torch.nn.functional as F

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
        if spec_L.layers == spec.layers:
            keep_blocks = None
        elif spec_L.layers % spec.layers == 0:          # integer stride (pair B mapping)
            stride = spec_L.layers // spec.layers
            keep_blocks = list(range(0, spec_L.layers, stride))
        else:                                            # evenly spaced (e.g. 32 -> 24)
            keep_blocks = sorted({round(i * (spec_L.layers - 1) / (spec.layers - 1))
                                  for i in range(spec.layers)})
            assert len(keep_blocks) == spec.layers, "depth mapping produced duplicates"
        kw = dict(heads=spec_L.heads, head_dim_small=spec.d_model // spec.heads,
                  mlp_small=4 * spec.d_model, rotary_pct=rotary_pct,
                  heads_small=spec.heads if spec.heads != spec_L.heads else None,
                  keep_blocks=keep_blocks)
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
    ap.add_argument("--resume", action="store_true",
                    help="resume from data/m5/ckpt/<init><tag>.pt if present (for long/interruptible runs)")
    ap.add_argument("--distill", action="store_true",
                    help="add a frozen-teacher (the large donor) KD loss; complements any init arm")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.distill:
        cfg["distill"] = True
    if args.seed is not None:
        cfg["seed"] = args.seed
    # Seeds vary the data draw too (the pipeline is otherwise deterministic):
    # each seed trains on a disjoint stream offset; arms within a seed share it.
    cfg["skip_docs"] = cfg["skip_docs"] + cfg["seed"] * 25_000
    run = make_run(f"m5_{args.init}{args.tag}", cfg)
    dev = run.device
    tok = load_tokenizer(cfg["small"])

    model = build_init(args.init, cfg, dev)
    # Construction scratch (donor weights, per-layer Sigma for the compensation solve) is
    # freed but stays in the caching allocator; release it so the training step's fp32
    # logits block fits on a 40GB card (hybrid_rs OOM'd in the first backward otherwise).
    import gc; gc.collect(); torch.cuda.empty_cache()
    model.train()

    # Optional knowledge-distillation lever: the donor is queried each step as a
    # frozen teacher (a training-time channel, orthogonal to the init arm). Off
    # by default; used only for the SP+distill / ours+distill comparison arms.
    teacher, distill_T, distill_a = None, cfg.get("distill_T", 2.0), cfg.get("distill_alpha", 0.5)
    if cfg.get("distill"):
        teacher = load_model(cfg["large"], device=dev).eval()
        teacher.requires_grad_(False)
        print(f"[m5-{args.init}] distill ON (teacher={cfg['large']}, T={distill_T}, alpha={distill_a})", flush=True)
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

    # Optional checkpointing for long convergence runs (survives crashes/interruptions).
    ckpt_every = cfg.get("ckpt_every_tokens", 0)
    ckpt_path = Path("data/m5/ckpt") / f"{args.init}{args.tag}.pt"

    def save_ckpt():
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "seen": seen, "curve": curve}, tmp)
        tmp.replace(ckpt_path)   # atomic

    if args.resume and ckpt_path.exists():
        st = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); seen = st["seen"]; curve[:] = st["curve"]
        next_log = ((seen // cfg["log_every_tokens"]) + 1) * cfg["log_every_tokens"]
        print(f"[m5-{args.init}] resumed at t={seen/1e6:.0f}M; fast-forwarding data stream...", flush=True)

    batches = pile_train_batches(tok, seq_len=cfg["seq_len"], batch_size=cfg["batch_size"],
                                 skip_docs=cfg["skip_docs"])
    if seen:  # replay the deterministic stream to the resume point (order preserved)
        for _ in range(seen // (cfg["seq_len"] * cfg["batch_size"])):
            next(batches)
        model.train()
    for step, batch in enumerate(batches):
        batch = batch.to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            out = model(batch, labels=batch)
            loss = out.loss
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(batch).logits
                # match next-token distributions (align with the CE label shift);
                # flatten tokens so batchmean averages per token, not per sequence
                V = out.logits.size(-1)
                s = out.logits[:, :-1].reshape(-1, V).float() / distill_T
                t = t_logits[:, :-1].reshape(-1, V).float() / distill_T
                kd = F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1),
                              reduction="batchmean") * (distill_T ** 2)
                loss = distill_a * loss + (1.0 - distill_a) * kd
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
            if ckpt_every and seen % ckpt_every < cfg["seq_len"] * cfg["batch_size"]:
                save_ckpt()
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
