#!/usr/bin/env python
"""Read-only audit of outputs_local/diagnostic_results.json (M0/M1/M2).

The v1.2.4 run had four construction defects and reported M2 with a negative
pooled R2 (about -0.061) and MAE about 0.110.  This verifier re-checks the
v1.2.5 construction fix and the v1.2.6 preregistration alignment without
fitting anything or writing any file:

  * 23 unique line feature rows, no zero-distance distinct-line pairs
  * GroupKFold(5) isolation: no same-fold neighbour, target line absent from
    every M0 source, neighbour pools drawn from training lines only
  * M1 RidgeCV fitted on all 62 pre-registered PCs
  * M2 kNN uses the pre-registered first 10 PCs
  * forcing every M2 neighbour weight to zero returns M0 exactly
  * all reported metrics are finite; M2 pooled R2 is positive, so the v1.2.4
    sign/construction bug cannot pass unnoticed
  * no holdout ModelID strings in the results (count only)
  * diagnostic_results.csv mirrors the JSON numbers

Exit 0 on PASS, 1 on failure.  Reads only; never writes.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs_local"
JSON_PATH = OUT / "diagnostic_results.json"
CSV_PATH = OUT / "diagnostic_results.csv"
EXPR_AUDIT = OUT / "expression_line_audit.json"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> int:
    for path in (JSON_PATH, CSV_PATH, EXPR_AUDIT):
        if not path.exists():
            print(f"FAIL missing required artifact: {path}")
            return 1

    res = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    text = JSON_PATH.read_text(encoding="utf-8")
    expr = json.loads(EXPR_AUDIT.read_text(encoding="utf-8"))
    holdout = [str(x) for x in expr.get("ml_holdout_ids", [])]

    check("version", res.get("version") == "v1.2.6", str(res.get("version")))
    check("label", "post-hoc, exploratory" in str(res.get("label", "")).lower(),
          str(res.get("label")))
    check("training lines", res.get("training_lines") == 23,
          str(res.get("training_lines")))
    check("M1 feature count", res.get("m1_n_pcs") == 62, str(res.get("m1_n_pcs")))
    pcs = res.get("pcs_used") or []
    check("M2 feature count", pcs == [f"pca_{i}" for i in range(10)], str(pcs))
    check("holdout count only", res.get("holdout_lines_excluded_count") == 7,
          str(res.get("holdout_lines_excluded_count")))
    check("no holdout ID list key", "holdout_modelids_excluded" not in res)
    check("NDCG note", bool(res.get("ndcg10_note")))
    check("Recall note", bool(res.get("recall10_note")))
    leaked = [mid for mid in holdout if mid in text]
    check("no holdout ModelID strings", not leaked, ", ".join(leaked) or "none")
    check("not withdrawn", "withdrawn" not in text.lower())

    audit = res.get("construction_audit") or {}
    pca = audit.get("pca_line_matrix") or {}
    check("PCA rows", pca.get("n_unique_feature_rows") == pca.get("n_lines") == 23,
          f"{pca.get('n_unique_feature_rows')}/{pca.get('n_lines')}")
    check("PCA distinct lines", pca.get("zero_distance_pairs") == 0,
          f"zero-distance pairs={pca.get('zero_distance_pairs')}")
    check("PCA minimum distance", finite(pca.get("min_pairwise_distance"))
          and float(pca["min_pairwise_distance"]) > 0,
          str(pca.get("min_pairwise_distance")))

    iso = audit.get("fold_isolation") or {}
    check("GroupKFold folds", iso.get("n_folds") == 5, str(iso.get("n_folds")))
    check("same-fold neighbours", iso.get("same_fold_neighbour_count") == 0,
          str(iso.get("same_fold_neighbour_count")))
    check("target line not in M0 source",
          iso.get("target_line_in_m0_source") is False,
          str(iso.get("target_line_in_m0_source")))
    check("M0 sources training-only",
          iso.get("m0_sources_are_training_lines_only") is True,
          str(iso.get("m0_sources_are_training_lines_only")))

    m1_fit = audit.get("m1_fit") or {}
    check("M1 RidgeCV fitted", m1_fit.get("ridge_fitted") is True)
    check("M1 fit features", m1_fit.get("n_features") == 62,
          str(m1_fit.get("n_features")))
    check("M1 per-fold alpha", len(m1_fit.get("per_fold_alpha") or []) == 5,
          str(m1_fit.get("per_fold_alpha")))

    zw = audit.get("zero_weight_sanity") or {}
    check("zero-weight rows", zw.get("rows_checked", 0) > 0,
          str(zw.get("rows_checked")))
    check("zero-weight equals M0", zw.get("exact_equal") is True
          and zw.get("max_abs_diff_vs_m0") == 0,
          str(zw.get("max_abs_diff_vs_m0")))

    metrics = res.get("metrics") or {}
    for name in ("M0", "M1", "M2"):
        m = metrics.get(name) or {}
        vals = [m.get("perline_spearman_mean"), m.get("pooled_r2_vs_truth"),
                m.get("mae_vs_truth")]
        check(f"{name} metrics finite", all(finite(v) for v in vals),
              ", ".join(str(v) for v in vals))
    m0, m1, m2 = metrics.get("M0", {}), metrics.get("M1", {}), metrics.get("M2", {})
    check("M1 per-line equals M0",
          finite(m1.get("perline_spearman_mean"))
          and math.isclose(float(m1["perline_spearman_mean"]),
                           float(m0.get("perline_spearman_mean", float("nan"))),
                           rel_tol=0.0, abs_tol=1e-12),
          f"{m1.get('perline_spearman_mean')} vs {m0.get('perline_spearman_mean')}")
    check("M1 pooled R2 is fitted, not copied",
          finite(m1.get("pooled_r2_vs_truth"))
          and not math.isclose(float(m1["pooled_r2_vs_truth"]),
                               float(m0.get("pooled_r2_vs_truth", float("nan"))),
                               rel_tol=0.0, abs_tol=1e-12),
          f"{m1.get('pooled_r2_vs_truth')} vs {m0.get('pooled_r2_vs_truth')}")
    check("M2 pooled R2 positive (v1.2.4 bug guard)",
          finite(m2.get("pooled_r2_vs_truth"))
          and float(m2["pooled_r2_vs_truth"]) > 0,
          str(m2.get("pooled_r2_vs_truth")))
    check("M2 MAE below old bug value",
          finite(m2.get("mae_vs_truth")) and float(m2["mae_vs_truth"]) < 0.10,
          str(m2.get("mae_vs_truth")))

    pb = (res.get("paired_bootstrap_lines_2000") or {}).get("m2_minus_m0") or {}
    ci = pb.get("ci_95") or []
    check("M2-M0 paired CI present", len(ci) == 2 and all(finite(v) for v in ci),
          str(ci))

    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8", newline="")))
    by_model = {r.get("model"): r for r in rows}
    check("CSV has M0/M1/M2", set(by_model) >= {"M0", "M1", "M2"},
          ",".join(by_model))
    for name, key in (("M0", "perline_spearman_mean"), ("M1", "pooled_r2"),
                      ("M1", "mae"), ("M2", "pooled_r2"), ("M2", "mae")):
        row = by_model.get(name) or {}
        csv_val = row.get(key)
        json_key = {"perline_spearman_mean": "perline_spearman_mean",
                    "pooled_r2": "pooled_r2_vs_truth",
                    "mae": "mae_vs_truth"}[key]
        json_val = metrics.get(name, {}).get(json_key)
        ok = finite(csv_val) and finite(json_val) and math.isclose(
            float(csv_val), float(json_val), rel_tol=0.0, abs_tol=1e-9)
        check(f"CSV {name} {key} mirrors JSON", ok,
              f"{csv_val} vs {json_val}")

    failed = [c for c in CHECKS if not c[1]]
    for name, ok, detail in CHECKS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f": {detail}" if detail else ""))
    if failed:
        print(f"\nDIAGNOSTIC AUDIT FAILED ({len(failed)}/{len(CHECKS)} checks)")
        return 1
    print(f"\nDIAGNOSTIC AUDIT PASS ({len(CHECKS)}/{len(CHECKS)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
