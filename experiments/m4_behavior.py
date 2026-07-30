#!/usr/bin/env python
"""M4 test 5: does the learned residual correction improve the assembled model?

Builds two 410M-architecture models and compares wikitext perplexity:
  - operator: matrices = Ŵ (fitted-operator projection); biases/LNs/embeddings
    from the real 410M (an ablation isolating the 6 matrix types)
  - operator+R: matrices = Ŵ + R(Ŵ) with the MLP-512 patch predictor trained on
    the 18 non-held-out layers (corrections on those layers are partly memorized;
    the 6 held-out layers are genuine predictions — noted in the report).

    uv run python experiments/m4_behavior.py --config configs/m2.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.assemble import build_model  # noqa: E402
from lib.data import DATA_DIR  # noqa: E402
from lib.eval import wikitext_perplexity  # noqa: E402
from lib.models import extract_weights, load_model, load_tokenizer  # noqa: E402
from lib.residuals import (PATCH, TYPES, patch_dataset, predict_patches,  # noqa: E402
                           train_patch_predictor, unpatch)
from lib.utils import load_config, make_run  # noqa: E402

from m4_predictor import TEST_LAYERS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = make_run("m4_behavior", cfg)
    small, dev = cfg["small"], run.device
    tok = load_tokenizer(small)

    model = load_model(small, device=dev)
    W = extract_weights(model)          # start from real small: biases/LN/emb stay
    del model
    torch.cuda.empty_cache()
    n_layers = 1 + max(l for l, _ in W)

    # per-type: Ŵ, plus the training rows for R
    What, Dstd, Xs, Ys, Ls = {}, {}, [], [], []
    for ti, t in enumerate(TYPES):
        D = torch.load(DATA_DIR / "deltas" / f"{t}.pt", weights_only=True).to(dev, torch.float32)
        WS = torch.stack([W[(l, t)] for l in range(n_layers)]).to(dev)
        What[t], Dstd[t] = WS - D, D.std()
        X, Y, li = patch_dataset(What[t] / What[t].std(), D / Dstd[t], ti, dev)
        Xs.append(X); Ys.append(Y); Ls.append(li)
    X, Y, li = torch.cat(Xs), torch.cat(Ys), torch.cat(Ls)
    train = ~torch.isin(li, torch.tensor(TEST_LAYERS, device=dev))
    R = train_patch_predictor(X[train], Y[train], hidden=512, seed=0)

    def ppl_with(matrices) -> float:
        Wv = dict(W)
        for t in TYPES:
            for l in range(n_layers):
                Wv[(l, t)] = matrices[t][l]
        m = build_model(Wv, small, ln_source="small", device=dev)
        p = wikitext_perplexity(m, tok)
        del m
        torch.cuda.empty_cache()
        return p

    results = {"operator": ppl_with(What)}
    print(f"[m4-beh] operator     ppl={results['operator']:,.1f}")

    corrected = {}
    for ti, t in enumerate(TYPES):
        Xt, _, _ = patch_dataset(What[t] / What[t].std(), What[t], ti, dev)  # Y unused
        pred = predict_patches(R, Xt).view(n_layers, -1, PATCH * PATCH) * Dstd[t]
        O, I = What[t].shape[1], What[t].shape[2]
        corrected[t] = What[t] + torch.stack([unpatch(pred[l], O, I) for l in range(n_layers)])
    results["operator_plus_R"] = ppl_with(corrected)
    print(f"[m4-beh] operator+R   ppl={results['operator_plus_R']:,.1f}")

    run.save_json("behavior.json", {"small": small, "test_layers": TEST_LAYERS,
                                    "wikitext103_ppl": results})
    print(f"[m4-beh] saved -> {run.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
