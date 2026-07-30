"""Deterministic perplexity on the wikitext-103 validation set. [M3]

Standard strided sliding-window evaluation: every token is scored exactly once,
with overlapping context from the previous window, so the number is
reproducible and comparable across models.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


def _eval_text(dataset: str) -> str:
    """One long evaluation document. wikitext103 = the standard; c4 = out-of-domain
    (different corpus from both the Pile training stream and wikitext)."""
    if dataset == "wikitext103":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
        return "\n\n".join(line for line in ds["text"] if line.strip())
    if dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        from itertools import islice
        return "\n\n".join(ex["text"] for ex in islice(ds, 500))
    raise ValueError(dataset)


@torch.no_grad()
def wikitext_perplexity(model, tokenizer, *, seq_len: int = 1024,
                        stride: int = 512, device=None,
                        max_tokens: int | None = None,
                        dataset: str = "wikitext103") -> float:
    """Compute the model's perplexity on wikitext-103 validation text.

    Perplexity is exp(average negative log-likelihood per token) — lower means
    the model predicts the text better. We join the split's non-empty lines
    into one long document, tokenize it once, then slide a window of `seq_len`
    tokens across it in steps of `stride`. Consecutive windows overlap; inside
    each window the overlapping prefix is context only (labels masked to -100)
    and just the new tokens are scored, so every token counts exactly once.
    Expects a model already in eval mode; `device` defaults to wherever the
    model's weights live. Loss is computed in float32 for numeric stability.
    """
    ids = tokenizer(_eval_text(dataset), return_tensors="pt").input_ids   # (1, n_tokens)
    if device is None:
        device = next(model.parameters()).device

    n = ids.shape[1] if max_tokens is None else min(ids.shape[1], max_tokens)
    total_nll, total_tokens, prev_end = 0.0, 0, 0
    for begin in tqdm(range(0, n, stride), desc=f"{dataset} ppl"):
        end = min(begin + seq_len, n)
        window = ids[:, begin:end].to(device)
        labels = window.clone()
        labels[:, : prev_end - begin] = -100    # already scored by the previous window
        logits = model(window).logits.float()
        shift_logits = logits[:, :-1, :].flatten(0, 1)          # position t predicts t+1
        shift_labels = labels[:, 1:].flatten()
        total_nll += F.cross_entropy(shift_logits, shift_labels,
                                     ignore_index=-100, reduction="sum").item()
        total_tokens += int((shift_labels != -100).sum())
        prev_end = end
        if end == n:
            break
    return math.exp(total_nll / total_tokens)
