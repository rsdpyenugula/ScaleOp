# External review notes

A running log of outside feedback on the ScaleOp research direction, and what we
decided to do about each point. Kept so decisions are traceable — nothing gets
silently dropped or silently adopted.

> **Sync:** the HTML tracker `project_docs/status.html` mirrors the status of
> every item below. Update both together.

## Status legend

| Status | Meaning |
|---|---|
| ✅ Accepted & applied | Agreed with; already reflected in the plan/code. |
| 👍 Accepted — planned | Agreed with; scheduled for a later milestone/phase (not done yet). |
| 🤔 Under consideration | Open decision; needs a call from us before it changes anything. |
| ⚠️ Conflict — needs decision | Contradicts a current assumption; can't proceed until resolved. |
| 🔭 Deferred to future phase | Interesting but out of scope now; parked for Phase 2/3. |
| ❌ Declined | Considered and not adopting (with reason). |

---

## R1 — external framing review, 2026-07-27: reframe around cross-family scaling-trajectory transfer

**Summary of the feedback.** Reduce the project to one question: *"Can scaling
trajectories learned from a checkpoint-rich family (Pythia) generalize to
checkpoint-poor families (e.g., Qwen)?"* Train the operator on Pythia; at
inference apply it using only another family's final checkpoint. Success = better
init / less fine-tuning than existing methods.

| # | Point | Status | Notes / decision |
|---|---|---|---|
| R1.1 | Adopt the crisp one-sentence "trajectories generalize across families" framing for the paper story | 🤔 Under consideration | Genuinely clearer than the original "are inter-size residuals structured?" mechanism framing. Likely adopt as the *narrative*, but see R1.2 first. |
| R1.2 | "Trajectory" axis is ambiguous: **size** (70M→1.4B, final checkpoints) vs **training-time** (Pythia's ~154 intermediate checkpoints) | ✅ Resolved — SIZE axis (2026-07-29) | The roadmap (Paper 1 within-family size conversion → Paper 2 bigger pairs → Paper 3 other families) is entirely size-based. Training-time trajectories deferred indefinitely. |
| R1.3 | Cross-family transfer Pythia→Qwen at the weight level | 🔭 Deferred to future phase | Highest-risk claim: different d_model/layers/heads, tokenizer, data, and architecture (Qwen RMSNorm+SwiGLU vs Pythia LayerNorm+GELU) mean **no shared basis** to apply a weight-space operator. Likely only viable in an aligned/normalized representation, not raw weights. Park as the Phase 2 headline; de-risk before committing. |
| R1.4 | Use a within-family transfer test as a cheap proxy for "is the structure family-level, not Pythia-specific?" | ✅ Accepted & applied | Already in the plan as **M4's transfer test** (operator trained on 1.4B→410M, tested on held-out 410M→160M). If it fails there, cross-*family* is hopeless — a valuable early kill-switch. |
| R1.5 | Success criterion = better initialization / less fine-tuning than existing methods | ✅ Accepted & applied | Matches the existing plan (**M5** killer-baseline: recovery curves vs subcloning at matched compute). |

**Assessment (2026-07-27):** keep the R1.1 framing for the paper,
but stage it — Phase 1 stays within-family size conversion (buildable, de-risked),
R1.4 is the cheap transfer proxy, and R1.3 (Qwen) becomes Phase 2 only after
deciding weight-space vs aligned-representation transfer. Resolve R1.2 first.

**Open actions:** decide R1.2 (which trajectory) → then update `Phase1.md`
framing and, if training-time is chosen, its model/data assumptions.

---

## R2 — external code review, 2026-07-27: of `lib/activations.py`

**Summary.** Rated the activation-capture implementation ~8.5–9/10; clean and
efficient. Flagged robustness/clarity improvements and two "future-proofing"
points. Correction on one item: our model loads in **fp32**, so the `.float()`
call was a no-op (not an extra copy) — reducing with `dtype=float32` is a clarity
/ bf16-safety change, and produces byte-identical caches.

| # | Point | Status | Notes / decision |
|---|---|---|---|
| R2.1 | Clear `captured` each batch + assert every hook fired | ✅ Accepted & applied | Cheap guard against a silently-missing layer. |
| R2.2 | `captured.pop(i)` to free each layer's GPU tensor as processed | ✅ Accepted & applied | Lowers peak GPU memory during the post-forward loop. |
| R2.3 | Drop redundant `.float()`; reduce in fp32 via `mean(dtype=float32)` | ✅ Accepted & applied | Model is fp32 so it wasn't a cost; refactor is clarity + bf16-safety. Caches byte-identical. |
| R2.4 | Use an explicit seq counter for the CKA subset (not `start < cka_subset`) | ✅ Accepted & applied | `seen` counter; clearer intent, robust to batching changes. |
| R2.5 | Abstract layer access beyond GPT-NeoX (`model.gpt_neox.layers`) | 🔭 Deferred to future phase | Phase 1 is entirely Pythia/GPT-NeoX; a multi-arch layer now is speculative. Ties to R1.3 (cross-family, Phase 2). Left a comment marking the assumption. |
| R2.6 | Handle edge cases `cka_subset == 0` / empty tokens (`torch.cat([])`) | ❌ Declined (for now) | Those inputs never occur (fixed 10k corpus, we control all callers). Guarding them is speculative defensive code; revisit if a caller ever needs it. |
| R2.7 | Hooks hold all layer activations on GPU until the forward completes | ✅ Noted — no action | Inherent to capturing every layer; partially mitigated by R2.2. |
| R2.8 | Style: `def stack_layers` over a named lambda; typed hook; explicit `hidden` var | ✅ Accepted & applied | Also silences the assign-lambda lint. |

---

## R3 — Methodology-standards audit, 2026-07-29

**Goal:** verify our choices against industry/academic standards so
reviewers can't dismiss findings on setup grounds. Grounded in: Weight Subcloning
(arXiv:2312.09299), Sheared LLaMA (arXiv:2310.06694), LiGO (arXiv:2303.00980).

Already standard (no action): wikitext-103 sliding-window ppl; Pile training data
(Pythia-native); AdamW(0.9, 0.95)+cosine+warmup+wd 0.1+clip 1.0; LR 3e-4 (Pythia-410M's
own); CKA/Procrustes/effective-rank; matched-budget recovery framing.

| # | Point | Status | Notes |
|---|---|---|---|
| R3.1 | lm-eval harness tasks (lambada, arc_easy, hellaswag, piqa) missing | 👍 Accepted — planned | Run on the four trained M5 checkpoints before arXiv (meaningful only post-training). |
| R3.2 | Single seed for the race | 👍 Accepted — planned | ≥3 seeds on the unseen 410M→160M pair; state seed counts explicitly. |
| R3.3 | Subcloning implementation fidelity vs the reference recipe | 🤔 Under consideration | Verify importance scoring + any weight-rescaling step in arXiv:2312.09299; else label ours "subcloning-style (magnitude selection)". |
| R3.4 | seq_len 1024 (vs Pythia 2048) and 32k-token batches | ✅ Accepted & applied | Documented as deviations in M5.md; fine for relative comparisons. |
| R3.5 | Position vs LiGO (learned linear growth, mirror direction) and Sheared LLaMA (learned-mask pruning + continued pretraining) | 👍 Accepted — planned | Required related-work anchors for M6; hybrid's compensation cites pruning-with-reconstruction lineage; claim the setting, not the primitive. |
