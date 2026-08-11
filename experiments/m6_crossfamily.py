"""M6: cross-family compatibility probe.

Paper 1 showed conversion value rides on the *representation* channel: within a
family, a smaller sibling's activations are largely a linear image of the
larger's (held-out ridge R^2 = 0.84) even though the weights are not. This script
asks the prerequisite question for converting ACROSS families: does that linear
relationship survive when donor and target are different architectures trained on
different data with different tokenizers?

If cross-family R^2 is close to the within-family number, the representation
channel is open and weight-space conversion is worth attempting. If it collapses,
selection-based conversion cannot work across those families and the bottleneck is
the representation geometry itself, not the conversion algorithm.

Two obstacles make this comparison non-trivial, and both are handled here:
  1. Different tokenizers. The same text becomes different token sequences of
     different lengths, so activations are not row-aligned per token. We compare
     ONE VECTOR PER DOCUMENT (mean-pooled over the sequence), which is
     tokenization-invariant: row i of both matrices summarizes the same text.
  2. Different depths. Layer i of a 24-block donor is not layer i of a 16-block
     target, so we compare at matched *relative* depth (fraction through the
     stack) and also report the all-pairs CKA matrix for the full picture.

Usage (donor/target are HF ids or Pythia suite keys):
    uv run python experiments/m6_crossfamily.py \
        --donor EleutherAI/pythia-1.4b --target Qwen/Qwen2.5-0.5B
    uv run python experiments/m6_crossfamily.py --donor 1.4b --target 410m  # control
"""
from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from lib import alignment  # noqa: E402
from lib.data import _load_text_stream  # noqa: E402
from lib.models import MODEL_SUITE  # noqa: E402
from lib.utils import make_run  # noqa: E402


def block_list(model):
    """The transformer blocks of an HF causal LM, whatever the family calls them.

    lib/activations.py hardcodes `model.gpt_neox.layers`; cross-family work needs
    this indirection (GPT-NeoX, Llama/Qwen/Mistral/OLMo, and GPT-2/Falcon differ).
    """
    for path in ("model.layers", "gpt_neox.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj, path
    raise AttributeError(f"could not locate transformer blocks on {type(model).__name__}")


def mean_pooled(model_id: str, texts: list[str], *, seq_len: int, device, batch_size: int = 8):
    """Per-document, per-layer mean-pooled hidden states: (n_layers, n_docs, d).

    Each model tokenizes the SAME texts with its OWN tokenizer, so the token grids
    differ; mean-pooling over the sequence collapses that difference away.
    """
    name = MODEL_SUITE[model_id].name if model_id in MODEL_SUITE else model_id
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
    blocks, path = block_list(model)
    print(f"[m6] {name}: {len(blocks)} blocks via .{path}, d_model={model.config.hidden_size}")

    captured: dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(_m, _inp, out):
            captured[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]
    pooled: list[list[torch.Tensor]] = [[] for _ in range(len(blocks))]
    try:
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start:start + batch_size]
                enc = tok(chunk, return_tensors="pt", padding="max_length", truncation=True,
                          max_length=seq_len)
                ids = enc.input_ids.to(device)
                mask = enc.attention_mask.to(device).unsqueeze(-1)  # ignore pad in the mean
                captured.clear()
                model(ids, attention_mask=enc.attention_mask.to(device))
                for i in range(len(blocks)):
                    h = captured[i].float() * mask
                    pooled[i].append((h.sum(1) / mask.sum(1).clamp(min=1)).cpu())
    finally:
        for h in handles:
            h.remove()
    out = torch.stack([torch.cat(p) for p in pooled])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", required=True, help="HF id or Pythia suite key")
    ap.add_argument("--target", required=True, help="HF id or Pythia suite key")
    ap.add_argument("--n-docs", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--train-frac", type=float, default=0.75, help="rows used to fit the map")
    args = ap.parse_args()

    run = make_run(f"m6_{args.donor.split('/')[-1]}_to_{args.target.split('/')[-1]}", vars(args))
    dev = run.device

    # One fixed text sample, shared by both models (tokenized separately by each).
    stream, provenance = _load_text_stream()
    texts = [t for t in islice((t for t in stream if t and t.strip()), args.n_docs)]
    print(f"[m6] {len(texts)} docs from {provenance}")

    X = mean_pooled(args.donor, texts, seq_len=args.seq_len, device=dev)   # (Ld, n, dd)
    Y = mean_pooled(args.target, texts, seq_len=args.seq_len, device=dev)  # (Lt, n, dt)
    Ld, n, _ = X.shape
    Lt = Y.shape[0]
    n_train = int(n * args.train_frac)

    # Matched relative depth: target layer j <- donor layer round(j * (Ld-1)/(Lt-1)).
    ridge, proc, pairs = [], [], []
    for j in range(Lt):
        i = round(j * (Ld - 1) / max(Lt - 1, 1))
        ridge.append(alignment.ridge_r2(X[i], Y[j], n_train))
        proc.append(alignment.procrustes_r2(X[i], Y[j], n_train))
        pairs.append((i, j))
    cka = alignment.cka_matrix(X, Y, device=dev).tolist()

    r_mean = sum(ridge) / len(ridge)
    p_mean = sum(proc) / len(proc)
    run.save_json("crossfamily.json", {
        "donor": args.donor, "target": args.target, "provenance": provenance,
        "n_docs": n, "seq_len": args.seq_len, "donor_layers": Ld, "target_layers": Lt,
        "depth_pairs": pairs, "ridge_r2": ridge, "procrustes_r2": proc,
        "ridge_r2_mean": r_mean, "procrustes_r2_mean": p_mean,
        "ridge_r2_best": max(ridge), "cka_matrix": cka,
    })
    print(f"[m6] {args.donor} -> {args.target}: ridge R^2 mean {r_mean:.3f} "
          f"(best layer {max(ridge):.3f}), procrustes {p_mean:.3f}")
    print(f"[m6] within-family reference from Paper 1 (1.4b->410m): ridge R^2 ~0.84")
    print(f"[m6] -> {run.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
