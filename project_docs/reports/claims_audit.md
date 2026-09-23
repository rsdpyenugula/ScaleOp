# Claims Audit — "Wiring Beats Blending"

Traces every quantitative claim in `paper/main.tex` (abstract) and `paper/sections/*.tex`
to its primary source on disk. Every number below was verified by opening the raw
JSON under `project_docs/results/**/*.json` (not the markdown reports) and comparing.

**Status legend**
- **VERIFIED** — matches the raw result JSON (to displayed precision).
- **SECONDARY** — value exists only in a markdown report, no raw JSON found.
- **MISMATCH** — differs from the recomputed/raw value (both shown).
- **UNSOURCED** — no source file found.
- **CONFIG** *(extension)* — architecture/protocol constant (Pythia config or training
  hyperparameter); external ground truth, not expected in `results/**/*.json`. Not a
  data-integrity issue; listed for completeness.
- **METRIC-MIX** *(flag, appended)* — number is fine, but it is *compared* across the two
  perplexity metrics (t=0 quick-eval on 50k tokens vs. full strided WikiText-103). Task-required flag.

All primary-pair analysis files carry `"large":"1.4b","small":"410m"`; pair-B carries `410m→160m`.
Paths are relative to repo root. Timestamped dirs elided to the parent for brevity where unambiguous.

---

## A. Abstract (one row per number)

| # | claim (short) | paper location | value | primary source file | status |
|---|---|---|---|---|---|
| A1 | primary conversion pair | abstract | 1.4B→410M | `results/m2_maps/.../maps_r2.json` (`large/small`) | VERIFIED |
| A2 | representations align (ridge R²) | abstract | R²=0.84 | `results/m2_maps/20260727_231935/maps_r2.json` (`ridge_r2_mean`=0.8441) | VERIFIED |
| A3 | hybrid_rs final ppl (mean±std) | abstract | 84.0±1.8 | `m5_hybrid_rs{,_s1,_s2}/20260731*/curve.json` (exclude `20260803*` 1B runs) | VERIFIED (83.955±1.80) |
| A4 | subclone_rs final ppl (mean±std) | abstract | 89.7±3.7 | `results/m5_subclone_rs{,_s1,_s2}/.../curve.json` | VERIFIED (89.675±3.73) |
| A5 | paired within-seed wins | abstract/§7 | 3/3 | `m5_hybrid_rs*` vs `m5_subclone_rs*` per-seed `ppl_full` | VERIFIED (83.0<86.4, 86.0<93.7, 82.8<88.9) |
| A6 | speedup over from-scratch @30M | abstract/§7 | 18× | `m5_random`(1519.03)/`m5_hybrid_rs`(83.02) → 18.3 | VERIFIED |
| A7 | held at 2× training context | abstract/§7 | 2048 | `results/m5_extended_eval/.../extended_eval.json` (`wikitext_ppl_ctx2048`) | VERIFIED |
| A8 | held-out depth-dominated pair, 3 seeds | abstract/§7 | 3 seeds | `m5_{hybrid,subclone,random}_b_s{0,1,2}/.../curve.json` | VERIFIED |
| A9 | 1B convergence parity | abstract/§7/§8 | 40.0 vs 40.0 | `m5_{hybrid_rs,subclone_rs}/20260803_052406/curve.json` | VERIFIED (39.956/39.959) |
| A10 | 1B random final | abstract/§7/§8 | 57.4 | `m5_random/20260803_052406/curve.json` | VERIFIED (57.411) |

## B. §3 Setup (architecture & protocol constants)

| # | claim (short) | paper location | value | primary source file | status |
|---|---|---|---|---|---|
| S1 | primary widths: residual 2048→1024 | §3 | 2048→1024 | `results/m2_weights/.../weights.json` (`shape_large/small` = [50304,2048]/[50304,1024]) | VERIFIED |
| S2 | primary: head dim 128→64, MLP 8192→4096, 24L/16H | §3 | — | (Pythia config) | CONFIG |
| S3 | held-out pair 410M→160M: 24→12 blocks, 16→12 heads, ~25% width | §3/§7 | — | `results/m4_transfer/.../transfer.json` names `pair_B=[410m,160m]`; dims are config | CONFIG |
| S4 | scale pair 6.9B→1.4B | §3/§7 | Table 10 | `m5_{subclone_rs,hybrid,hybrid_rs,random}_c/.../curve.json` | VERIFIED |
| S5 | frozen corpus 10,000×128 tok; first 500 subset | §3 | 10000/500 | (protocol; not in results JSON) | CONFIG |
| S6 | AdamW β(.9,.95), wd 0.1, warmup 100, clip 1.0 | §3 | — | (protocol) | CONFIG |
| S7 | token budget 30M / 100M / 1B | §3/§7/§8 | 30M/100M/1B | curve `tokens` in resolved_config.yaml | VERIFIED |

## C. §4 Representations align, parameters do not

| # | claim (short) | paper location | value | primary source file | status |
|---|---|---|---|---|---|
| R1 | CKA diagonal mean | §4 / Table 1 | 0.883 | `results/m2_cka/20260727_220648/cka.json` (24×24 diag mean = 0.88317) | VERIFIED |
| R2 | layers with best match on diagonal | §4 | 12% | `m2_cka/.../cka.json` — 3/24 = **12.5%** (paper truncates to 12%) | VERIFIED (note truncation) |
| R3 | ridge mean held-out R² | §4 / Table 1 | 0.844 | `m2_maps/.../maps_r2.json` (`ridge_r2_mean`=0.84409) | VERIFIED |
| R4 | Procrustes mean held-out R² | §4 / Table 1 | 0.717 | `m2_maps/.../maps_r2.json` (`procrustes_r2_mean`=0.71660) | VERIFIED |
| R5 | ridge−Procrustes gap | §4 | ~0.13 | 0.844−0.717 = 0.127 | VERIFIED |
| R6 | held-out variance captured | §4 | 84% | = ridge 0.844 | VERIFIED |
| R7 | train / held-out split | §4 | 8k / 2k | `m2_maps/.../maps_r2.json` (`n_train`=8000) | VERIFIED |
| R8 | ridge & Procrustes dip at layers | §4 | 3–5 | `maps_r2.json` arrays: ridge min idx 3–5 (.797/.802/.793), procrustes min idx 3 (.632) | VERIFIED |
| R9 | input embedding ridge R² | §4 / Table 1 | 0.248 | `results/m2_weights/.../weights.json` (`EMB_IN.ridge_r2`=0.24807) | VERIFIED |
| R10 | output embedding ridge R² | §4 / Table 1 | 0.386 | `m2_weights/.../weights.json` (`EMB_OUT.ridge_r2`=0.38637) | VERIFIED |
| R11 | embedding ridge range | §4 | 0.25–0.39 | from R9/R10 | VERIFIED |
| R12 | input embedding CKA | Table 1 | 0.512 | `weights.json` (`EMB_IN.cka_before`=0.51210) | VERIFIED |
| R13 | output embedding CKA | Table 1 | 0.678 | `weights.json` (`EMB_OUT.cka_before`=0.67762) | VERIFIED |
| R14 | input embedding Procrustes R² | Table 1 | 0.167 | `weights.json` (`EMB_IN.procrustes_r2`=0.16684) | VERIFIED |
| R15 | output embedding Procrustes R² | Table 1 | 0.242 | `weights.json` (`EMB_OUT.procrustes_r2`=0.24242) | VERIFIED |

## D. §5 Why dense projection fails

Source for P1–P12: `results/m3_project/20260728_011729/projection_error.json`.
Source for P15–P19/P23: `results/m3_eval/20260728_010702/eval.json`.

| # | claim (short) | paper location | value | primary source file | status |
|---|---|---|---|---|---|
| P1 | Q projection (target-free) | Table 2 (proj) | 1.471 | `projection_error.json` Q.projection=1.47065 | VERIFIED |
| P2 | K projection | Table 2 | 1.591 | K.projection=1.59103 | VERIFIED |
| P3 | V projection | Table 2 | 1.390 | V.projection=1.39028 | VERIFIED |
| P4 | O projection | Table 2 | 1.724 | O.projection=1.72398 | VERIFIED |
| P5 | MLP_UP projection | Table 2 | 1.668 | MLP_UP.projection=1.66825 | VERIFIED |
| P6 | MLP_DOWN projection | Table 2 | 1.723 | MLP_DOWN.projection=1.72256 | VERIFIED |
| P7 | Q operator (fitted) | Table 2 | 0.658 | Q.operator=0.65837 | VERIFIED |
| P8 | K operator | Table 2 | 0.758 | K.operator=0.75825 | VERIFIED |
| P9 | V operator | Table 2 | 0.804 | V.operator=0.80363 | VERIFIED |
| P10 | O operator | Table 2 | 0.795 | O.operator=0.79484 | VERIFIED |
| P11 | MLP_UP operator | Table 2 | 0.683 | MLP_UP.operator=0.68347 | VERIFIED |
| P12 | MLP_DOWN operator | Table 2 | 0.686 | MLP_DOWN.operator=0.68570 | VERIFIED |
| P13 | fitted operator range | §5/§6 | 0.66–0.80 | min Q 0.658→0.66, max V 0.804→0.80 | VERIFIED |
| P14 | weight variance explainable | §5/§6 | ~30% | derived: 1−relerr ≈ 0.20–0.34 for best-fit operators | VERIFIED (derived; "~30%" is a loose 1−relerr reading) |
| P15 | real 1.4B reference ppl | Table 3 (anchor) | 11.5 | `eval.json` `real_large`=11.4553 | VERIFIED |
| P16 | real 410M upper anchor ppl | Table 3 | 15.6 | `eval.json` `real_small`=15.5695 | VERIFIED |
| P17 | random-init 410M lower anchor | Table 3 | 65,962 | `eval.json` `random_init`=65962.19 (strided) | VERIFIED |
| P18 | projected, LN copied from 410M | Table 3 | 181,242 | `eval.json` `projected_ln_small`=181242.53 (strided) | VERIFIED |
| P19 | projected, LN projected from 1.4B | Table 3 | 10¹³ | `eval.json` `projected_ln_project`=1.008e13 | VERIFIED |
| P20 | neutral-LN projected quick-ppl | §5 footnote | ~12.3k | `results/m5_projection/20260729_191359/curve.json` t=0 `ppl`=12323.09 | VERIFIED |
| P21 | neutral-LN vs 181,242 row | §5 footnote | ~15× | 181242/12323 = 14.7 | VERIFIED — **METRIC-MIX** (12.3k quick vs 181,242 strided; footnote *does* label 12.3k "quick-perplexity", distinction preserved) |
| P22 | neutral-LN vs random init | §5 footnote | ~5× | 65962/12323 = 5.35 | VERIFIED — **METRIC-MIX** (quick 12.3k vs strided random 65,962) |
| P23 | projected-LN variant explodes | §5 | 10¹³ | = P19 | VERIFIED |

## E. §6 Post-operator residuals are noise

Source for E1–E10: `results/m4_residuals/20260729_013805/residual_tests.json`.
Source for E11–E13,E17: `results/m4_predictor/20260729_021957/predictor.json`.
Source for E14–E15: `results/m4_transfer/20260729_022039/transfer.json`.
Source for E18–E20: `results/m4_behavior/20260729_022516/behavior.json`.

| # | claim (short) | paper location | value | primary source file | status |
|---|---|---|---|---|---|
| E1 | Q Δ erank | Table 4 (erank) | 771.3 | `residual_tests.json` Q.erank=771.303 | VERIFIED |
| E2 | K Δ erank | Table 4 | 784.0 | K.erank=783.973 | VERIFIED |
| E3 | V Δ erank | Table 4 | 801.2 | V.erank=801.160 | VERIFIED |
| E4 | O Δ erank | Table 4 | 792.2 | O.erank=792.216 | VERIFIED |
| E5 | MLP_UP Δ erank | Table 4 | 947.2 | MLP_UP.erank=947.243 | VERIFIED |
| E6 | MLP_DOWN Δ erank | Table 4 | 934.8 | MLP_DOWN.erank=934.785 | VERIFIED |
| E7 | attn shuffled/Gaussian erank | Table 4 | 824.0–824.2 | `residual_tests.json` erank_shuffle/gauss (824.05–824.16) | VERIFIED |
| E8 | MLP shuffled/Gaussian erank | Table 4 | 989.4 | erank_shuffle/gauss = 989.35–989.38 | VERIFIED |
| E9 | Δ concentration below controls | §6/Test 1/Fig | 3–6% | computed **2.79%–6.41%** below shuffle (Q 6.41%, V 2.79%) | VERIFIED (band slightly wider than stated "3–6%") |
| E10 | cross-layer cosine of Δ | §6/Test 2 | ≈±0.0002 | `residual_tests.json` `xlayer_cos` ∈ [5e-6, 2.0e-4]; max |·| = Q 2.0e-4 | VERIFIED (all ≈0; ±0.0002 is the max magnitude) |
| E11 | linear reduction (held-out) | Table 5 | −0.0004 | `predictor.json` linear.err_reduction=−0.000406 | VERIFIED |
| E12 | MLP-512 reduction (held-out) | Table 5 | −0.0003 | mlp512.err_reduction=−0.000338 | VERIFIED |
| E13 | gap CI₉₅ ≈ [−0.0000,−0.0000], p≥0.99 | §6/Table 5 | p 0.99/1.00 | linear p=0.992, mlp512 p=1.0; CIs ∈ [−3.8e-5, −1.9e-6] | VERIFIED |
| E14 | transfer reduction (410M→160M) | Test 4/Table 5 | −0.0004/−0.0003, p=1.00 | `transfer.json` linear=−0.000429, mlp512=−0.000263, p=1.0 | VERIFIED |
| E15 | transfer shuffle control | Table 5 | −0.0004/−0.0002 | `transfer.json` shuffle linear=−0.000408, mlp512=−0.000235 | VERIFIED |
| E16 | pair-B operators fit range | §6/Test 4 | 0.56–0.74 | **no raw JSON** — only `project_docs/reports/M4.md` line 37 | **SECONDARY** |
| E17 | train / held-out layer split | §6/Test 3 | 18 / 6 | `predictor.json` `test_layers`=[3,7,11,15,19,23] (6 held-out) | VERIFIED |
| E18 | operator-projected 410M ppl | §6 | 35,212 | `behavior.json` `operator`=35212.37 | VERIFIED |
| E19 | operator+correction ppl (~2% move) | §6 | 34,373 | `behavior.json` `operator_plus_R`=34372.78; (35212−34373)/35212=2.4% | VERIFIED |
| E20 | operator beats random (35k vs 66k) | §6 | 35k/66k | `behavior.json` 35,212 vs `eval.json` `random_init` 65,962 | VERIFIED |

## F. §7 Ablation ladder (Table: `tab:ladder`, primary pair, 30M)

Zero-shot = curve `ppl` at t=0 (quick-eval); Final = last `ppl_full` (strided).

| # | claim (short) | paper location | value | primary source file | status |
|---|---|---|---|---|---|
| L1 | random zero-shot | Table 6 (ladder) | ~66,000 | `m5_random/20260729_201149/curve.json` t0=66247.20 | VERIFIED |
| L2 | random final | ladder | 1,519.0 | `m5_random/.../curve.json` `ppl_full`=1519.03 | VERIFIED |
| L3 | projection zero-shot | ladder | 12,323 | `m5_projection/20260729_191359/curve.json` t0=12323.09 | VERIFIED |
| L4 | projection final | ladder | 1,054.5 | `m5_projection/.../curve.json` `ppl_full`=1054.53 | VERIFIED |
| L5 | subclone zero-shot | ladder | 11,509 | `m5_subclone/20260729_181940/curve.json` t0=11509.41 | VERIFIED |
| L6 | subclone final (30M) | ladder | 355.8 | `m5_subclone/20260729_181940/curve.json` `ppl_full`=355.80 | VERIFIED |
| L7 | subclone_iso zero-shot | ladder | 26,669 | `m5_subclone_iso/20260731_025148/curve.json` t0=26669.36 | VERIFIED |
| L8 | subclone_iso final | ladder | 208.2 | `m5_subclone_iso/.../curve.json` `ppl_full`=208.24 | VERIFIED |
| L9 | hybrid zero-shot | ladder | 9,664 | `m5_hybrid/20260729_222250/curve.json` t0=9664.54 | VERIFIED |
| L10 | hybrid final (30M) | ladder | 114.1 | `m5_hybrid/20260729_222250/curve.json` `ppl_full`=114.09 | VERIFIED |
| L11 | subclone_rs zero-shot | ladder | 61,912 | `m5_subclone_rs/20260731_003106/curve.json` t0=61911.74 | VERIFIED |
| L12 | subclone_rs final | ladder | 86.4 | `m5_subclone_rs/.../curve.json` `ppl_full`=86.37 | VERIFIED |
| L13 | hybrid_rs zero-shot | ladder | 18,459 | `m5_hybrid_rs/20260731_034607/curve.json` t0=18458.52 | VERIFIED |
| L14 | hybrid_rs final | ladder | 83.0 | `m5_hybrid_rs/.../curve.json` `ppl_full`=83.017 | VERIFIED |
| L15 | subclone vs random speedup | §7 | 4.3× | 1519.03/355.80 = 4.27 | VERIFIED |
| L16 | projection vs from-scratch | §7 | 1.4× | 1519.03/1054.53 = 1.44 | VERIFIED |
| L17 | full-ladder speedup | §7 | 18× | 1519.0/83.0 = 18.30 | VERIFIED |
| L18 | compensation moves zero-shot | §7 | 11,509→9,664 | L5, L9 | VERIFIED |
| L19 | compensation moves endpoint | §7 | 355.8→114.1 (3.1×) | 355.80/114.09 = 3.12 | VERIFIED |
| L20 | iso rescale zero-shot/endpoint | §7 | 11,509→26,669 / 355.8→208.2 | L5,L7 / L6,L8 | VERIFIED |
| L21 | subclone_rs zero-shot collapse | §7 | 61,912 = 5.4× worse | 61912/11509 = 5.38 | VERIFIED |
| L22 | hybrid_rs dominates both axes | §7 | 18,459<61,912 / 83.0<86.4 | L13,L11 / L14,L12 | VERIFIED |

## G. §7 Seeded verdict (Table: `tab:seeded`, primary pair, 30M)

| # | claim (short) | paper location | value | primary source file | status |
|---|---|---|---|---|---|
| T1 | subclone_rs seeds 0/1/2 | Table 7 | 86.4/93.7/88.9 | `m5_subclone_rs`(86.37)/`_rs_s1`(93.71)/`_rs_s2`(88.94) `ppl_full` | VERIFIED |
| T2 | subclone_rs mean±std | Table 7 | 89.7±3.7 | recomputed 89.675 ± 3.73 | VERIFIED |
| T3 | hybrid_rs seeds 0/1/2 | Table 7 | 83.0/86.0/82.8 | `m5_hybrid_rs`(83.02)/`_rs_s1`(86.03)/`_rs_s2`(82.82) `ppl_full` | VERIFIED |
| T4 | hybrid_rs mean±std | Table 7 | 84.0±1.8 | `m5_hybrid_rs{,_s1,_s2}/20260731*/curve.json` (exclude 1B) | VERIFIED (83.955±1.80) |
| T5 | hybrid_rs worst beats subclone_rs best | §7 | 86.0 < 86.4 | T3 max 86.03 < T1 min 86.37 | VERIFIED |
| T6 | paired wins | §7 | 3/3 | per-seed (see A5) | VERIFIED |

## H. §7 Hardening — extended eval (Table: `tab:extended`, primary, 30M)

Source: `results/m5_extended_eval/20260729_233557/extended_eval.json`. Each row = all 6 metrics matched.

| # | claim (short) | paper location | value | primary source file | status |
|---|---|---|---|---|---|
| X1 | hybrid: C4 90.7 / wk@2048 109.8 / lam 0.076 / arc 0.313 / hella 0.266 / piqa 0.555 | Table 8 | as shown | `extended_eval.json` hybrid (90.709/109.775/0.07646/0.31313/0.26568/0.55495) | VERIFIED |
| X2 | subclone: 162.7/346.3/0.005/0.308/0.263/0.548 | Table 8 | as shown | subclone (162.734/346.304/0.00543/0.30808/0.26270/0.54842) | VERIFIED |
| X3 | projection: 380.5/1210/0.000/0.272/0.258/0.539 | Table 8 | as shown | projection (380.524/1209.783/0.0/0.27231/0.25752/0.53917) | VERIFIED |
| X4 | random: 518.8/1788/0.000/0.264/0.256/0.532 | Table 8 | as shown | random (518.773/1788.251/0.0/0.26389/0.25553/0.53210) | VERIFIED |
| X5 | real 410M: 21.0/14.5/0.516/0.519/0.337/0.667 | Table 8 | as shown | real_410m (20.998/14.473/0.51640/0.51894/0.33718/0.66703) | VERIFIED |
| X6 | LAMBADA hybrid vs subclone | §7 | 14× | 0.07646/0.005434 = 14.07 | VERIFIED |

## I. §7 Longer budget (100M) & held-out pair (Table: `tab:unseen`)

| # | claim (short) | paper location | value | primary source file | status |
|---|---|---|---|---|---|
| H1 | hybrid 100M final | §7 | 62.3 | `m5_hybrid/20260730_050004/curve.json` `ppl_full`=62.25 | VERIFIED |
| H2 | subclone 100M final | §7 | 103.1 | `m5_subclone/20260730_012010/curve.json` `ppl_full`=103.10 | VERIFIED |
| H3 | hybrid crosses subclone-100M endpoint | §7 | ~38M tok (~2.6×) | `m5_hybrid/20260730_050004/curve.json` quick-ppl crosses 103.1 at ~37.4M; 100/38=2.63 | VERIFIED — METRIC-MIX (crossing on quick-ppl vs subclone full endpoint; footnoted as schedule-confounded/single-seed) |
| U1 | hybrid(b) seeds 0/1/2 | Table 9 | 117.5/113.6/108.5 | `m5_hybrid_b_s0/s1/s2/.../curve.json` (117.489/113.573/108.485) | VERIFIED |
| U2 | hybrid(b) mean±std | Table 9 | 113.2±4.5 | recomputed 113.182 ± 4.51 | VERIFIED |
| U3 | subclone(b) seeds 0/1/2 | Table 9 | 115.0/121.6/118.9 | `m5_subclone_b_s0/s1/s2` (114.968/121.578/118.921) | VERIFIED |
| U4 | subclone(b) mean±std | Table 9 | 118.5±3.3 | recomputed 118.489 ± 3.33 | VERIFIED |
| U5 | random(b) seeds 0/1/2 | Table 9 | 1,464.4/1,571.5/1,479.6 | `m5_random_b_s0/s1/s2` (1464.413/1571.512/1479.623) | VERIFIED |
| U6 | random(b) mean±std | Table 9 | 1,505.2±57.9 | recomputed 1505.183 ± 57.94 (task hint "1505±58" ✓) | VERIFIED |
| U7 | transfer vs from-scratch every seed | §7 | ~13× | per-seed random/transfer ratios 12.4–13.8× | VERIFIED |
| U8 | hybrid/subclone tie (±1σ overlap) | §7 | overlap | 113.2±4.5 vs 118.5±3.3 overlap | VERIFIED |
| U9 | hybrid pair-B zero-shot | §7 | ≈30k | `m5_hybrid_b_s0/.../curve.json` t0=30444.18 | VERIFIED |
| U10 | hybrid width-pair zero-shot | §7 | 9.7k | `m5_hybrid/.../curve.json` t0=9664.54 | VERIFIED |
| U11 | 100M seed-0 hybrid vs subclone | §7 | 77.1 vs 74.4 (3.6%) | `m5_hybrid_b100`(77.15)/`m5_subclone_b100`(74.44); (77.1−74.4)/74.4=3.6% | VERIFIED |
| U12 | pair-B C4 hybrid vs subclone | §7 | 91.1 vs 94.2 | `results/m5_extended_eval_b_s0/.../extended_eval.json` (91.127/94.210) | VERIFIED |
| U13 | pair-B ctx-2048 hybrid vs subclone | §7 | 111.8 vs 110.2 | extended_eval_b_s0 (111.789/110.211) | VERIFIED |
| U14 | pair-B LAMBADA hybrid vs subclone | §7 | 11.0% vs 10.7% | extended_eval_b_s0 (0.11023/0.10673) | VERIFIED |
| U15 | pair-B random C4 / LAMBADA | §7 | 514 / 0 | extended_eval_b_s0 random (514.366/0.0) | VERIFIED |
| U16 | real-160M anchor LAMBADA | §7 | 35.4% | extended_eval_b_s0 real_160m (0.35377) | VERIFIED |

## J. §7 Compute accounting & scale

| # | claim (short) | paper location | value | primary source file | status |
|---|---|---|---|---|---|
| C1 | compensation moments pass | §3/§7 | ~75s / 1,000 seqs | (protocol timing; not in results JSON) | CONFIG |
| C2 | per-run wall time / throughput | §7 | ~55 min / ~9.4k tok/s | 30,015,488 tok ÷ 9,400 = 53 min (self-consistent); budget VERIFIED via curve `maxtok` | CONFIG (rate) / VERIFIED (30M) |
| C3 | scale pair 6.9B→1.4B results | §7 Table 10 | 572/776/1213/1413 | `m5_{subclone_rs,hybrid,hybrid_rs,random}_c/.../curve.json` | VERIFIED |

## K. §7 / §8 1B convergence (2026-08-03 re-run)

Source: `project_docs/results/m5_{hybrid_rs,subclone_rs,random}/20260803_052406/curve.json`.
Checkpoints: `artifacts/m5_1B_checkpoints/{hybrid_rs,subclone_rs,random}.pt`.
**Note:** 1B and 30M runs share the `m5_hybrid_rs` prefix — disambiguate by timestamp
(`20260803_052406` = 1B; `20260731_*` = 30M).

| # | claim (short) | paper location | value | primary source file | status |
|---|---|---|---|---|---|
| K1 | hybrid_rs 1B final, seed 0 | App. `tab:conv1b` | 40.0 | `ppl_full`=39.9565 | VERIFIED |
| K2 | subclone_rs 1B final, seed 0 | App. `tab:conv1b` | 40.0 | `ppl_full`=39.9587 | VERIFIED |
| K3 | random 1B final | §7/§8 | 57.4 | `ppl_full`=57.4109 | VERIFIED |
| K4 | transfer vs scratch @1B | §7 | ~1.4× | 57.41/40.3 = 1.42× | VERIFIED |
| K5 | 1B checkpoints on disk | release | 3 files | `artifacts/m5_1B_checkpoints/*.pt` | VERIFIED |
| K6 | hybrid_rs 1B, 3 seeds (2026-09-22 AWS p4d re-run, seeds 1–2) | abstract/§7 | 40.3±0.3 | `results_aws/m5_hybrid_rs_1b_s{1,2}/*/curve.json` `ppl_full`=40.40, 40.47 (+K1) → mean 40.28 sd 0.28 | VERIFIED |
| K7 | subclone_rs 1B, 3 seeds | abstract/§7 | 40.3±0.5 | `results_aws/m5_subclone_rs_1b_s{1,2}/20260921_231345/curve.json` (resumed-from-600M runs; the `165004` dirs are the OOM'd first attempts) `ppl_full`=40.08, 40.93 (+K2) → mean 40.32 sd 0.53 | VERIFIED |
| K8 | 1B extended eval, seeds 0–2 (C4 / wk@2048 / 4 tasks) | §7, `tab:conv1b` | as tabled; wk@2048 hybrid 74.2±4.8 vs subclone 67.3±0.9 (hybrid worse 3/3); C4 44.46 vs 44.41 | `results_aws/m5_extended_eval_1b_s{0,1,2}/*/extended_eval.json` (s0 run 2026-09-23 on the Spark from `artifacts/m5_1B_checkpoints`; includes random) | VERIFIED |
| K9 | 1B seed-1/2 weights | release | 4 files | `s3://de-aiml-scaleop-662022802750/conv_reseed/checkpoints/{hybrid_rs,subclone_rs}_1b_s{1,2}.pt` (1.5 GiB each; not on the Mac) | VERIFIED |

## L. §7 Information-matched control (Table: `tab:actsel`, primary pair, 30M; Kumar review #2)

Source: `scratch/m5_act_results/m5_{subclone_rs_act,hybrid_rs_act}_s{0,1,2}/*/curve.json`
(Spark, isolated `~/projects/ScaleOp_p1` at master `a76741b`, 2026-09-22/23; `configs/m5.yaml`,
batch 32, seeds 0–2 = same data draws as `tab:seeded`). Code: `lib/subclone.py::select_indices(moments=...)`.

| # | claim (short) | paper location | value | primary source file | status |
|---|---|---|---|---|---|
| L1 | subclone_rs_act per seed | `tab:seeded` | 86.9 / 89.8 / 85.5 | `ppl_full` per dir | VERIFIED |
| L2 | hybrid_rs_act per seed | `tab:seeded` | 81.7 / 81.4 / 77.6 | `ppl_full` per dir | VERIFIED |
| L3 | subclone_rs_act mean±std | `tab:actsel` | 87.4±2.2 | mean 87.38 sd 2.18 | VERIFIED |
| L4 | hybrid_rs_act mean±std | `tab:actsel` | 80.2±2.3 | mean 80.21 sd 2.25 | VERIFIED |
| L5 | information effect, no comp | §7 | 2.3, 2/3 seeds | 86.4→86.9 (−0.5), 93.7→89.8, 88.9→85.5 | VERIFIED |
| L6 | compensation effect, norm / act | §7 | 5.7 / 7.2, 3/3 each | (3.4, 7.7, 6.1) / (5.2, 8.4, 7.9) | VERIFIED |
| L7 | hybrid_rs beats subclone_rs_act, paired | abstract/§1/§7/§9 | 3/3 | 83.0<86.9, 86.0<89.8, 82.8<85.5 | VERIFIED |
| L8 | "recovers less than half of that gap" (abstract) | abstract | 2.3 / 5.7 = 0.40 | information effect over the hybrid_rs–subclone_rs gap | VERIFIED |

*(Rows K6–K9 and section L added 2026-09-23; the summary counts below predate them.)*

---

## Summary counts

Total quantitative claim rows audited: **137**.

| status | count |
|---|---|
| VERIFIED | 131 |
| SECONDARY | 1 |
| MISMATCH | 0 |
| UNSOURCED | 0 |
| CONFIG (external constants, not data issues) | 5 |
| — of the VERIFIED, additionally flagged METRIC-MIX | 3 (P21, P22, H3) |

Per-section VERIFIED tally: A 10/10 · S 3/7 (4 CONFIG) · R 15/15 · P 23/23 · E 19/20 (1 SECONDARY) ·
L 22/22 · T 6/6 · X 6/6 · H+U 19/19 · J 2/3 (1 CONFIG) · K 5/5.

**Run disambiguation:** `m5_hybrid_rs/20260803_052406` is the 1B convergence run;
30M seeded runs use `20260731_*` timestamps under the same prefix.

## Pre-arXiv fixes required (everything not cleanly VERIFIED)

1. ~~**[MISMATCH] hybrid_rs seeded mean**~~ **RESOLVED** — paper now shows 84.0±1.8.

2. **[SECONDARY] pair-B operator fit range "0.56–0.74" (§6, Test 4)** has no raw JSON.
   `results/m4_transfer/.../transfer.json` stores only the predictor reductions, not the
   pair-B operator relative errors; the 0.56–0.74 figures live only in
   `project_docs/reports/M4.md` (line 37). Either regenerate/commit the raw pair-B
   `projection_error.json` (410M→160M), or soften/cite the source before arXiv.

3. **[METRIC-MIX — disclosed, verify wording] §5 footnote "~15×/~5× better" (P21/P22)**
   compares neutral-LN **quick-perplexity** (12.3k, `m5_projection` t=0) against
   **strided** WikiText-103 numbers (181,242 and 65,962 from `m3_eval`). The footnote
   already labels 12.3k as "quick-perplexity," so the distinction is preserved — but the
   ratios themselves cross metrics. Confirm this is acceptable or add a half-sentence caveat.

4. **[METRIC-MIX — footnoted] §7 "hybrid crosses subclone's 100M endpoint at ~38M" (H3)**
   measures hybrid's per-checkpoint **quick-eval** ppl against subclone's **full** 100M
   endpoint (103.1). Interpolated crossing is ~37.4M (rounds to ~38M; 2.6× ✓). Already
   footnoted as schedule-confounded/single-seed; no change needed, noted for completeness.

5. **[Minor wording] §6/Test 1 "3–6% below controls" (E9)** — actual per-type deficits are
   **2.79%–6.41%** (V=2.79%, Q=6.41%), slightly outside the stated band on both ends.
   Consider "≈3–6%" or "3–7%".

6. **[Minor wording] §4 "12% of layers" (R2)** — exact value is 3/24 = **12.5%**
   (truncated to 12%). Optional.

No **UNSOURCED** items: every quantitative result claim traces to a raw JSON, except the
one SECONDARY (item 2) and the CONFIG constants (Pythia architecture / training-protocol
values, which by nature do not live in `results/**/*.json`).
