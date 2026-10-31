# Diagnostic archive: v1.2.4 misconstructed M2

- `diagnostic_results_v1.2.4_MISCONSTRUCTED.json`
- `diagnostic_results_v1.2.4_MISCONSTRUCTED.csv`

These are the v1.2.4 outputs. The statistic wiring was corrected in v1.2.4
(metrics against measured AUC; paired bootstrap over lines), but the M2
construction was still wrong:

1. `idx_of[L]` (a line rank 0..22) was used as a row index into a 23,658-row
   drug-line matrix. Only 3/23 mappings hit the intended line; duplicate rows
   produced zero distances and 1e9 weights.
2. The borrowed quantity was the neighbour's within-line-centred AUC, not the
   pre-registered M0 residual `AUC(d,L2) - M0_fold(d)`.
3. There was no `GroupKFold(5)` loop: per-drug means and the neighbour pool used
   all 23 lines, including the line being predicted and same-fold held-out lines.
4. M1 was never fitted; its CSV row copied M0 and `m1_equals_m0...` was hardcoded.

Superseded by the v1.2.5 fold-honest construction. Kept for audit history only.
