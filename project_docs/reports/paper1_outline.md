# Paper 1 outline — "Wiring Beats Blending" (working title)

Format base: the ICLR-accepted template used by our two closest neighbors —
LiGO (ICLR'23) and Sheared LLaMA (ICLR'24) — plus the standard accepted-paper
structure (funnel intro → explicit contribution bullets → method → controlled
experiments → limitations). Target: arXiv now; ICLR/NeurIPS main or an
efficiency workshop as venue. 9 pages main text.

## Title candidates
1. *Wiring Beats Blending: What Transfers Between Transformer Sizes — and What Doesn't*
2. *Size Conversion in Transformer Families: Representations Align, Parameters Don't*
3. *Selection + Compensation: Strong Initializations for Downscaling Pretrained Transformers*

## Abstract (formula: problem → gap → what we did → headline numbers → release)
Model families train every size from scratch. Can a pretrained large model be
converted into a smaller sibling? We characterize the 1.4B→410M conversion in
the Pythia family end-to-end: (i) representations align strongly across sizes
(ridge R²=0.84) while parameters align weakly; (ii) dense weight projection is
functionally destructive — provably not an assembly artifact — because blending
breaks rotary/per-head/GELU/LayerNorm structure; (iii) after the best-fit linear
operator, weight residuals are statistically indistinguishable from noise under
shuffle controls; (iv) therefore conversion value lives in *initialization*: in
a matched-budget recovery race, selection ("subcloning") beats dense blending
3×, and our **selection + closed-form least-squares compensation** beats
subcloning a further **3.1×** (wikitext ppl 114 vs 356 at 30M tokens; 13× better
than from-scratch), holding out-of-domain (C4) and at 2× training context.
Code, checkpoints, and the frozen evaluation corpus released.

## 1. Introduction (funnel: broad → thesis → contributions)
- Cost motivation: every family size = a separate pretraining run (GPUs/energy).
- The question: what actually transfers between sizes of the same family?
- Thesis: structure lives in *representations*, not *parameters*; conversion
  must respect wiring; value shows up as initialization, not zero-shot.
- **Contribution bullets (the accepted-paper formula, 4 max):**
  1. A controlled characterization: representations align (CKA/R²), parameters
     don't — with the LayerNorm-mismatch pitfall isolated.
  2. A mechanically-verified diagnosis of why dense projection fails
     (identity-reconstruction test → bit-exact; blame is on basis mixing).
  3. Null result with controls: post-operator residuals ≈ noise (spectral,
     consistency, predictor, transfer tests; bootstrap CIs).
  4. A winning method: rotary-frequency-matched selection + closed-form LS
     compensation → 3.1× over subcloning at matched compute, ~0 extra cost.

## 2. Related work (anchors reviewers expect — R3.5)
- Downscaling: Weight Subcloning (2312.09299); Sheared LLaMA (2310.06694);
  structured pruning + LS reconstruction lineage (our compensation's ancestry).
- Upscaling: Net2Net, bert2BERT, LiGO (2303.00980) — learned linear growth
  (mirror direction; consistent with our projection-beats-random).
- Representation similarity: CKA (Kornblith), stitching, model alignment.
- Positioning sentence: components exist; the *setting* (family size
  conversion), the rotary-matched selection, and the controlled
  characterization are new.

## 3. Setup  — *(write from: M0.md, M1.md — internal refs, not paper text)*
Pythia suite; primary pair 1.4B→410M (same depth/heads — isolates width);
held-out pair 410M→160M (depth+heads change). Frozen 10k×128 Pile corpus;
determinism; QKV re-fuse test. Hardware + exact budgets (compute accounting).

## 4. Representations align, parameters don't  — *(write from: M2.md)*
CKA heatmap (saturated middle band — known limitation, motivates R²); ridge
0.84 vs Procrustes 0.72 per layer; embeddings weight-space R² 0.25–0.39.
Figure: cka_heatmap, maps_r2.

## 5. Why dense projection fails  — *(write from: M3.md)*
Target-free projection (activation-PCA residual basis + shared SVD internal
bases); residuals >1; ppl table incl. the LN-mismatch correction (181k → 12.3k
with neutral LNs); identity-reconstruction bit-exact ⇒ diagnosis: rotary /
per-head / GELU / per-feature-LN all break under mixing. Table: anchors ppl.

## 6. Residuals are noise  — *(write from: M4.md)*
Four tests vs matched controls; predictor + transfer p≈1.0. Figure:
delta_spectra; predictor-vs-shuffle bars with CIs. (The paper's rigor core —
reviewers reward pre-registered-style controls.)

## 7. The recovery race — the payoff  — *(write from: M5.md)*
Four inits, identical budget/schedule/data (no target leakage); curves figure;
final table; hybrid method box (selection rules + `ls_compensate` equation);
hardening: [PLACEHOLDER extended-eval table: C4, ctx-2048, 4 tasks],
[PLACEHOLDER 100M persistence verdict + curves], [PLACEHOLDER unseen-pair
410M→160M ×3 seeds]. Compute accounting incl. init costs (~0).

## 8. Limitations & future work (the citation-farm section — explicit invites)
Single family (Pythia) → port to Llama/Qwen/DeepSeek/Kimi (adapter + per-arch
safe-blend analysis); 410M scale → 7B+ (Paper 2); budgets ≪ full recovery;
single/3 seeds; read-in compensation unexplored (LN renormalization); cross-
family operators likely need aligned representations, not raw weights.
Reduction-aware conversion (our planned extension): depth cuts are the
uncompensable axis — (a) block-rescale/distill deleted blocks into survivors,
(b) choose the reduction mix per parameter budget (prefer width over depth);
both fall out of the width-vs-depth boundary measured in §7.

## 9. Broader impact
Energy/cost reduction for model families; released artifacts.

## Figures checklist (per plan M6.2)
cka_heatmap ✓ · maps_r2 ✓ · projection/anchor ppl table ✓ · delta spectra ✓ ·
predictor-vs-shuffle bars w/ CIs (regen from json) · transfer result ✓ ·
m5_curves (4 runners) ✓ · [100M curves — pending] · [extended-eval table — pending]

## Release checklist
Repo public + README quickstart; checkpoints (4×30M + 2×100M) to HF; frozen
corpus recipe; `pip freeze`/uv.lock export; v0.1 tag (plan M6.3).

## Verdict templates (plan M6.1 — both pre-written; we landed on POSITIVE-hybrid)
- Positive (ours): "Conversion value is real but lives in initialization;
  structure-respecting selection + linear compensation dominates."
- Negative (kept for honesty about M4): "Zero-shot residual correction is not
  learnable at this granularity."
