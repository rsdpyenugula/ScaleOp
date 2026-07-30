"""Eval-corpus loading, tokenization, and the frozen token set. [M1]

Builds `data/eval_tokens.pt` once: N_SEQS fixed sequences of SEQ_LEN tokens,
tokenized with the Pythia tokenizer, with provenance recorded. This file is
frozen for the whole project so every model is probed on identical inputs. The
first CKA_SUBSET sequences are the subset used for full-token CKA analysis (M2).
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from .models import load_tokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_TOKENS_PATH = DATA_DIR / "eval_tokens.pt"

N_SEQS = 10_000       # number of eval sequences
SEQ_LEN = 128         # tokens per sequence
CKA_SUBSET = 500      # first N sequences reserved for full-token CKA (M2)


def _load_text_stream():
    """Yield raw text strings from the eval corpus, plus a provenance label.

    Priority (Plan §2): the Pile (Pythia's own training distribution, so
    in-distribution for representation analysis), falling back to C4 if the Pile
    isn't reachable. Both are streamed, so we only pull as much as we need.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        return (ex["text"] for ex in ds), "monology/pile-uncopyrighted:train"
    except Exception as e:  # network/gating/availability — log and fall back
        warnings.warn(f"Pile unavailable ({e}); falling back to C4")
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        return (ex["text"] for ex in ds), "allenai/c4:en:validation"


def build_eval_tokens(*, n_seqs: int = N_SEQS, seq_len: int = SEQ_LEN, force: bool = False) -> dict:
    """Build and freeze the eval token set (idempotent).

    If the frozen file already exists we just load it (it's the source of truth —
    never silently regenerated). Otherwise we stream text, tokenize it into one
    long token stream (documents separated by the end-of-sequence token), cut it
    into `n_seqs` blocks of `seq_len`, and save the tensor plus metadata.
    Returns {"tokens": LongTensor[n_seqs, seq_len], "meta": {...}}.
    """
    if EVAL_TOKENS_PATH.exists() and not force:
        return load_eval_tokens()

    tok = load_tokenizer("70m")  # the whole Pythia suite shares one tokenizer
    texts, provenance = _load_text_stream()
    need = n_seqs * seq_len

    buf: list[int] = []
    pbar = tqdm(total=need, desc="tokenizing", unit="tok")
    for text in texts:
        if not text.strip():
            continue
        ids = tok(text).input_ids
        buf.extend(ids)
        buf.append(tok.eos_token_id)  # mark document boundaries
        pbar.update(len(ids) + 1)
        if len(buf) >= need:
            break
    pbar.close()
    if len(buf) < need:
        raise RuntimeError(f"corpus too small: got {len(buf)} tokens, need {need}")

    tokens = torch.tensor(buf[:need], dtype=torch.long).view(n_seqs, seq_len)
    meta = {
        "provenance": provenance,
        "tokenizer": tok.name_or_path,
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "cka_subset": CKA_SUBSET,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": tokens, "meta": meta}, EVAL_TOKENS_PATH)
    return {"tokens": tokens, "meta": meta}


def load_eval_tokens() -> dict:
    """Load the frozen eval token set from disk."""
    return torch.load(EVAL_TOKENS_PATH, weights_only=True)


def pile_train_batches(tokenizer, *, seq_len: int, batch_size: int,
                       skip_docs: int = 10_000):
    """Endless stream of (batch_size, seq_len) token batches for fine-tuning. [M5]

    Streams the same corpus as the frozen eval set but SKIPS the first
    `skip_docs` documents — eval_tokens.pt was built from the stream's head, so
    the skip is the contamination guard. Documents are joined with EOS and cut
    into fixed-length blocks; identical call order gives identical batches, so
    every M5 arm trains on exactly the same data.

    Long runs can outlive the HF streaming connection (observed: httpx "client
    has been closed" ~34M tokens in). On any stream error we reopen and
    fast-forward past every document already consumed, so batch order — and
    therefore cross-arm comparability — is preserved exactly.
    """
    from itertools import islice

    buf: list[int] = []
    need = seq_len * batch_size
    consumed = 0
    for attempt in range(100):
        try:
            texts, _ = _load_text_stream()
            for text in islice(texts, skip_docs + consumed, None):
                consumed += 1
                if not text.strip():
                    continue
                buf.extend(tokenizer(text).input_ids)
                buf.append(tokenizer.eos_token_id)
                while len(buf) >= need:
                    chunk, buf = buf[:need], buf[need:]
                    yield torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len)
            return  # stream exhausted
        except Exception as e:  # network/stream death — reopen and fast-forward
            warnings.warn(f"train stream died after {consumed} docs ({e!r}); reopening "
                          f"(attempt {attempt + 1})")
