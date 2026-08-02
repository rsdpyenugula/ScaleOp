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
a matched-budget recovery race we decompose conversion into two independent
levers — least-squares **compensation** (function: best zero-shot) and
variance-preserving **rescale** (training dynamics: best endpoints) — whose
combination dominates the strongest subcloning variant on both axes and in
every seed (final ppl 84.0±1.8 vs 89.7±3.7, 3/3 paired wins; zero-shot 18.5k
vs 61.9k; 18× better than from-scratch at 30M tokens), holding out-of-domain
(C4), at 2× training context, and on a held-out depth-dominated pair (3 seeds).
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
hardening: extended-eval table (ordering holds on C4/ctx-2048/4 tasks; M5.md),
100M persistence (hybrid 62.3 vs subclone 103.1; ~2.6× token efficiency),
unseen pair ×3 seeds (hybrid 113.2±4.5 ≈ subclone 118.5±3.3, both ~13× over
random 1505±58 — the width-vs-depth compensation boundary). Compute accounting
incl. init costs (~0). 100M spot-check: unseen-pair tie persists (77.1 vs
74.4, leader flips between budgets ⇒ noise) — the boundary is budget-robust.

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
- **BEFORE flipping repo public: delete the Paper-2 branches from origin**
  (`paper2-anysize`, etc.). GitHub exposes ALL branches at once, so only `master`
  (Paper 1) should be public — or split Paper 1 into a fresh public repo.
- De-anonymize for arXiv (author/affiliation), drop line numbers, fill the
  reproducibility statement.
- Repo public + README quickstart; checkpoints (4×30M + 2×100M) to HF — the 1B
  checkpoints were lost to the AWS fault, so 30M/100M are the release set;
  frozen corpus recipe; `uv.lock` export; v0.1 tag.

## Style guide (extracted from the anchor papers' mechanics)

**Template/venue format:** ICLR LaTeX (both anchors are ICLR): single column,
9–10 pages main text, unlimited references/appendix, Reproducibility Statement
after the conclusion. NeurIPS variant adds the checklist + Broader Impact.

**Abstract (both anchors):** problem → named method → *numbers first* in the
results clause ("only 3% of compute" / "save up to 50%"). Ours leads with the
3.1×/13×/2.6× trio.

**Introduction (Sheared LLaMA move set):** motivation ¶ → *explicit italicized
research question* → approach ¶ → bulleted contributions (we have 4).

**Method math (LiGO move set):** density escalates — informal → review existing
operators with numbered eqs → our operators, each constraint justified (why
selection = the architecture's symmetry group; why compensation is the
closed-form optimum). Include an **Algorithm box** (selection scoring +
`ls_compensate`). Adopt their vocabulary: "structured sparse linear operators",
"factorize/decompose", "architecture-aware".

**Results conventions:**
- Tables: **bold best**, explicit budget column ("#tokens"), ± std where seeds
  exist, footnotes for caveats (cosine-horizon confound, single-seed cells).
- Curves: metric vs tokens AND wall-clock; **dashed horizontal line for the
  reference model** (real 410M/160M) — both anchors do this.
- Fairness: attribute every baseline's data/budget in-table (Sheared LLaMA
  footnote style); state that all arms share data order, schedule, budget.
- Savings phrasing: "reaches the same perplexity with X× fewer tokens" —
  LiGO's "reach the same performance" anchor sentence.

**Terminology map (ours → paper):** wires → units/dimensions · blend → dense
linear projection · selection → structured selection · race → matched-budget
continued pre-training · hybrid → **Compensated Selection (working name;
owner may rename)** · squeeze → size conversion.

## Writing plan (results-first order; ~5 working days)

1. **Figures/tables freeze** (0.5d): regenerate all figures at paper quality
   (dashed reference lines, budget axes), assemble every table with finals —
   nothing gets written against moving numbers.
2. **§4–7 results sections** (1.5d): drafted from M2–M5 reports (the numbers
   and verdicts are already written there; this is translation, not creation).
3. **§3 setup + Algorithm box + appendix configs** (0.5d): from M0/M1 +
   configs/*.yaml (LiGO-style reproducibility tables).
4. **§2 related work** (0.5d): anchors + pruning-with-reconstruction lineage +
   CKA/stitching; the R3.3 fidelity differences disclosed here.
5. **§1 intro + §8–9 + abstract LAST** (1d): abstract only after all numbers
   are frozen in tables.
6. **Review passes** (1d): owner rewrite/voice pass (the text should be the
   owner's), then a cold read against the reviewer-pin list (R3), then the
   reproducibility statement + release checklist.

Division of labor: draft skeleton + numbers + formatting = assistant;
**final prose voice, claims, and all sign-offs = owner** (also the honest
answer to any authorship question).

## Verdict templates (plan M6.1 — both pre-written; we landed on POSITIVE-hybrid)
- Positive (ours): "Conversion value is real but lives in initialization;
  structure-respecting selection + linear compensation dominates."
- Negative (kept for honesty about M4): "Zero-shot residual correction is not
  learnable at this granularity."
