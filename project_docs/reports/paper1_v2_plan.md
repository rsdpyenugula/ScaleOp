# Paper 1 — v2 strengthening plan

Two reviewer themes (structured pruning, distillation) reduce to a small,
concrete set of experiments plus honest prose. Key realization: our
`subclone_rs` arm **is** a structural-pruning init (magnitude selection), so
"SP + Distillation" = `subclone_rs + distill`. One code change (a distillation
loss) unlocks both experiments — no new selection code.

## Experiments (30M matched budget, Spark, 1 seed first, add 2 if promising)

Reuse the exact m5 harness (same data stream, schedule, budget) so results are
apples-to-apples with the published numbers. Teacher = the 1.4B donor, frozen,
queried each step.

| Arm | Init | Teacher in loop | Status |
|---|---|---|---|
| random (from scratch) | random | no | have (83.0 @30M) |
| SP (ours-selection) | subclone_rs | no | have (89.7 @30M) |
| **Ours** | hybrid_rs | no | have (84.0 @30M) |
| **SP + Distill** (SOTA recipe) | subclone_rs | **yes** | NEW |
| **Ours + Distill** (complementary) | hybrid_rs | **yes** | NEW |
| random + Distill (control) | random | yes | NEW (optional) |

**Metrics per arm:** final frozen-corpus perplexity (quality), wall-clock
(time), and teacher-forward overhead (effort: teacher params queried/step,
measured tok/s vs the no-teacher arms).

**Distillation loss:** `L = alpha*CE(student,data) + (1-alpha)*T^2*KL(student/T ||
teacher/T)`, next-token positions aligned with the CE shift. Defaults T=2,
alpha=0.5. Teacher & student share the Pythia vocab (50304), so logits align.

## What each outcome means (both are publishable)

- **Best:** Ours (no teacher) matches SP+Distill quality at a fraction of the
  time/effort, and Ours+Distill is best overall -> "cheap, competitive,
  complementary."
- **Realistic:** SP+Distill edges quality but at ~2x cost, and Ours+Distill >=
  SP+Distill -> our init still helps on top of the SOTA recipe.
- Either way it answers "you didn't compare to prune+distill."

## Prose fixes for v2 (honest, no results needed)

1. De-slop pass (em-dashes 68 -> ~15, italics 73 -> ~20, blunter topic lines).
2. Soften "train every size from scratch" -> "typically ... from scratch";
   name the exceptions (Sheared-LLaMA, Minitron) in the intro.
3. Scope statement vs distillation (weight-only, donor used once vs teacher
   throughout; complementary, not competing).
4. Soften abstract "we trace to ill-conditioning" -> "consistent with".
5. Frame contribution as *analysis of the transfer channel* (already 3/4
   contributions), not a new pruning method.
6. Future work: handling large donors — dimension-aware ridge + stronger
   pruning criteria (gradient/Hessian) for large-width donors.

## Sequencing

1. Implement distillation mode in `experiments/m5_train.py` (config-driven).
2. Smoke-test on Spark (teacher loads, loss finite, tok/s sane).
3. Run the NEW arms at 30M, 1 seed; pull results.
4. Build the time/effort/quality table + the prose fixes into v2; rebuild PDF +
   arXiv tar; post v2 (only after repo is public so "released" stays true).
