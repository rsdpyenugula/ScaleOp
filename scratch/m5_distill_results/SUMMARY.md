# Distillation vs prune+distill — 1.4B->410M, 30M tokens (Spark GB10, 2026-08-07)

subclone_rs = magnitude structured pruning init (the SP / Minitron/Sheared-LLaMA-style recipe base)
hybrid_rs   = Ours (compensated selection)
+distill    = frozen 1.4B teacher, KD loss (T=2, alpha=0.5)

| init         | b32 no-distill | b16 no-distill | b16 +distill |
|--------------|---------------:|---------------:|-------------:|
| subclone_rs  |           88.3 |           91.6 |         96.7 |
| hybrid_rs    |           82.9 |           88.3 |         83.2 |
| random       |         1521.1 |         1359.4 |       1311.3 |

Throughput: no-distill ~9.4k tok/s; +distill ~4.5k tok/s (teacher fwd every step, ~2x slower).
Clean batch-16 comparison: Ours beats SP with and without distill; distill helps Ours (88.3->83.2)
but hurts SP (91.6->96.7); Ours+distill (83.2) is best overall. Ordering Ours<SP holds at b16 and b32.
