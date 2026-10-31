# Diagnostic pre-correction archive

- `diagnostic_results_BUGGY.json` — pre-correction diagnostic results (superseded).
- `diagnostic_results_BUGGY.csv` — pre-correction diagnostic results (superseded).

DEVIATION: statistic implementation bug found post-run. The pre-correction diagnostic
computed metrics as M0-vs-M2 ranking agreements (and bootstrapped that agreement)
instead of computing per-line Spearman against measured AUC and paired M2-M0 differences
on measured AUC. The corrected results live in `outputs_local/diagnostic_results.json` and
`outputs_local/diagnostic_results.csv`. Superseded by the statistic fix-forward.
