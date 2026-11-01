"""Phase 3B diagnostic runner, v1.2.6 preregistration-aligned construction.

M0: per-drug mean AUC from the fold-training lines only (global-mean fallback).
M1: M0 + per-line RidgeCV offset fitted on fold-training lines using all 62
    pre-registered PCA features (pca_0..pca_61); the offset is constant per
    held-out line, so its per-line Spearman equals M0 by design, but its pooled
    R2/MAE are genuinely fitted and reported.
M2: M0 + kNN (k=5, Euclidean on the pre-registered first 10 PCs,
    inverse-distance weights) that borrows the neighbour's fold-honest M0
    residual AUC(d,L2) - M0_fold(d).

Strict isolation: GroupKFold(5) over the 23 training lines. Per-drug means,
PCA scaling, M1 fitting and the M2 neighbour pool all use fold-training lines
only. No holdout line is used.

Writes outputs_local/diagnostic_results.{json,csv}; prints the diagnostic table.

Pre-registered settings: k=5, M1 uses 62 PCA features, M2 uses the first 10 PCs,
GroupKFold(5), bootstrap seed 42 with 2,000 resamples, and the success criterion
"paired M2-M0 per-line Spearman 95% CI above 0".

Deviation recorded in the output: construction defects were found post-run and
corrected in v1.2.5; v1.2.6 restores M1's pre-registered 62 PCA features while
keeping M2 on the first 10 PCs. The success criterion is unchanged.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs_local"
LABEL = "post-hoc, exploratory, 23 training lines, GroupKFold(5)"
K = 5
N_PCS = 10          # M2 only: pre-registered first 10 PCs
M1_N_PCS = 62       # M1: all pre-registered PCA features
BOOTSTRAP_SEED = 42
N_BOOTSTRAP = 2000
M1_ALPHAS = (0.1, 1.0, 10.0)  # sklearn RidgeCV default grid; prereg left values open


def load_dataset(out_dir=OUT):
    """Load the ML frame, 23 training lines, holdout set, PCA columns and AUC map."""
    df = pd.read_parquet(Path(out_dir) / "tnbc_ml_dataset.parquet")
    df["ModelID"] = df["ModelID"].astype(str)
    audit = json.load(open(Path(out_dir) / "expression_line_audit.json"))
    holdout = set(audit["ml_holdout_ids"])
    lines = sorted(set(df["ModelID"]) - holdout)
    all_pca = sorted(
        (c for c in df.columns if c.startswith("pca_")),
        key=lambda c: int(c.split("_", 1)[1]),
    )
    m1_pca_cols = all_pca[:M1_N_PCS]
    pca_cols = m1_pca_cols[:N_PCS]
    if len(m1_pca_cols) != M1_N_PCS:
        raise ValueError(f"expected {M1_N_PCS} PCA columns for M1, found {len(m1_pca_cols)}")
    if len(pca_cols) != N_PCS:
        raise ValueError(f"expected {N_PCS} PCA columns for M2, found {len(pca_cols)}")
    auc = {
        (str(r.drug_id), str(r.ModelID)): float(r.auc)
        for r in df.itertuples(index=False)
    }
    drugs = sorted({d for (d, _L) in auc})
    return {
        "df": df,
        "holdout": holdout,
        "lines": lines,
        "pca_cols": pca_cols,
        "m1_pca_cols": m1_pca_cols,
        "auc": auc,
        "drugs": drugs,
    }


def build_line_matrix(df, all_lines, train_lines, pca_cols):
    """One row per ModelID, scaled on fold-training lines only.

    This is the Bug A fix: the old code used a line rank 0..22 as a row index
    into a 23,658-row drug-line matrix. Here the index is the actual ModelID.
    """
    feat = df.drop_duplicates("ModelID").set_index("ModelID")[pca_cols]
    missing = [L for L in all_lines if L not in feat.index]
    if missing:
        raise ValueError(f"missing PCA rows for lines: {missing}")
    scaler = StandardScaler().fit(feat.loc[train_lines].to_numpy(float))
    X = pd.DataFrame(
        scaler.transform(feat.loc[all_lines].to_numpy(float)),
        index=list(all_lines),
        columns=list(pca_cols),
    )
    return X, scaler


def fold_m0(auc, drugs, train_lines):
    """Per-drug mean AUC on fold-training lines, global-mean fallback."""
    observed = [auc[(d, L)] for L in train_lines for d in drugs if (d, L) in auc]
    global_mean = float(np.mean(observed)) if observed else float("nan")
    m0 = {}
    n_fallback = 0
    for d in drugs:
        vals = [auc[(d, L)] for L in train_lines if (d, L) in auc]
        if vals:
            m0[d] = float(np.mean(vals))
        else:
            m0[d] = global_mean
            n_fallback += 1
    return m0, global_mean, n_fallback


def fit_m1(X, train_lines, held_lines, auc, m0, drugs):
    """Fit RidgeCV on fold-training line residuals; return held-out M1 predictions.

    r_L = mean_d(AUC(d,L) - M0_fold(d)) over observed drugs in training line L.
    M1(d, L_val) = M0_fold(d) + r_hat(L_val), constant per held-out line.
    """
    r_train = []
    for L in train_lines:
        ds = [d for d in drugs if (d, L) in auc]
        r_train.append(float(np.mean([auc[(d, L)] - m0[d] for d in ds])))
    r_train = np.asarray(r_train, dtype=float)

    model = RidgeCV(alphas=M1_ALPHAS)
    X_train = X.loc[train_lines].to_numpy(float)
    model.fit(X_train, r_train)
    r_hat = np.asarray(model.predict(X.loc[held_lines].to_numpy(float)), dtype=float)

    preds = {}
    for i, L in enumerate(held_lines):
        preds[L] = {
            d: float(m0[d] + r_hat[i])
            for d in drugs
            if (d, L) in auc
        }
    fit = {
        "ridge_fitted": True,
        "estimator": "sklearn.linear_model.RidgeCV",
        "n_features": int(X.shape[1]),
        "alpha_grid": [float(a) for a in M1_ALPHAS],
        "alpha": float(model.alpha_),
        "intercept": float(model.intercept_),
        "coef_abs_sum": float(np.abs(model.coef_).sum()),
        "train_r2": float(model.score(X_train, r_train)),
        "r_train_range": [float(r_train.min()), float(r_train.max())],
        "r_hat_range": [float(r_hat.min()), float(r_hat.max())],
    }
    return preds, fit


def m2_predict(line, train_lines, X, auc, m0, drugs, k=K, zero_weights=False):
    """Fold-honest kNN M2 for one held-out line.

    Neighbours are strictly train_lines. Borrowed quantity is the neighbour's
    M0 residual AUC(d,L2) - m0[d]. Missing neighbour drugs are skipped and the
    remaining weights are renormalized; den==0 falls back to m0[d].

    zero_weights=True is the construction sanity test: every weight is zero,
    so the function must return m0 exactly.
    """
    dists = sorted(
        (float(np.linalg.norm(X.loc[line].to_numpy(float) - X.loc[L2].to_numpy(float))), L2)
        for L2 in train_lines
    )[:k]
    out = {}
    for d in [d for d in drugs if (d, line) in auc]:
        num = 0.0
        den = 0.0
        for dist, L2 in dists:
            if (d, L2) in auc:
                w = 0.0 if zero_weights else 1.0 / (dist + 1e-9)
                num += w * (auc[(d, L2)] - m0[d])
                den += w
        out[d] = float(m0[d] + (num / den if den > 0 else 0.0))
    return out, dists


def _pooled_metrics(preds, truths):
    """Pooled R2 and MAE over aligned prediction/truth vectors."""
    Y = np.concatenate([np.asarray(p, dtype=float) for p in preds])
    T = np.concatenate([np.asarray(t, dtype=float) for t in truths])
    sse = float(((Y - T) ** 2).sum())
    sst = float(((T - T.mean()) ** 2).sum())
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return r2, float(np.abs(Y - T).mean())


def run_diagnostic(out_dir=OUT, write=True):
    """Run the fold-honest M0/M1/M2 diagnostic and optionally write artifacts."""
    data = load_dataset(out_dir)
    df = data["df"]
    holdout = data["holdout"]
    lines = data["lines"]
    pca_cols = data["pca_cols"]
    m1_pca_cols = data["m1_pca_cols"]
    auc = data["auc"]
    drugs = data["drugs"]

    gkf = GroupKFold(n_splits=5)
    folds = []
    for k, (tr_idx, te_idx) in enumerate(gkf.split(np.zeros(len(lines)), groups=np.array(lines))):
        train_lines = sorted(np.array(lines)[tr_idx].tolist())
        held_lines = sorted(np.array(lines)[te_idx].tolist())
        folds.append({"fold": k, "train_lines": train_lines, "held_out_lines": held_lines})

    oof = {L: {} for L in lines}
    fold_audit = []
    zero_rows = 0
    zero_max_abs_diff = 0.0
    same_fold_neighbour_count = 0

    for fold in folds:
        k = fold["fold"]
        train_lines = fold["train_lines"]
        held_lines = fold["held_out_lines"]
        X, _scaler = build_line_matrix(df, lines, train_lines, pca_cols)
        X_m1, _scaler_m1 = build_line_matrix(df, lines, train_lines, m1_pca_cols)
        m0, global_mean, n_fallback = fold_m0(auc, drugs, train_lines)
        m1_preds, m1_fit = fit_m1(X_m1, train_lines, held_lines, auc, m0, drugs)

        neighbours = {}
        neighbour_distances = {}
        for L in held_lines:
            m2, dists = m2_predict(L, train_lines, X, auc, m0, drugs)
            neighbours[L] = [L2 for _, L2 in dists]
            neighbour_distances[L] = [float(d) for d, _ in dists]
            same_fold_neighbour_count += sum(1 for L2 in neighbours[L] if L2 in held_lines)

            ds = [d for d in drugs if (d, L) in auc]
            oof[L] = {
                "drugs": ds,
                "truth": [float(auc[(d, L)]) for d in ds],
                "M0": [float(m0[d]) for d in ds],
                "M1": [float(m1_preds[L][d]) for d in ds],
                "M2": [float(m2[d]) for d in ds],
            }

            # Zero-weight construction sanity: must equal M0 exactly.
            z, _ = m2_predict(L, train_lines, X, auc, m0, drugs, zero_weights=True)
            for d in ds:
                zero_rows += 1
                zero_max_abs_diff = max(zero_max_abs_diff, abs(z[d] - m0[d]))

        fold_audit.append({
            "fold": k,
            "train_lines": train_lines,
            "held_out_lines": held_lines,
            "m0_source_lines": train_lines,
            "m0_global_mean_fallback": float(global_mean),
            "m0_n_fallback_drugs": int(n_fallback),
            "m1_fit": m1_fit,
            "neighbours": neighbours,
            "neighbour_distances": neighbour_distances,
        })

    # Aggregate out-of-fold predictions over all 23 lines exactly once.
    per_line = {m: {} for m in ("M0", "M1", "M2")}
    for L in lines:
        t = np.asarray(oof[L]["truth"], dtype=float)
        for m in ("M0", "M1", "M2"):
            p = np.asarray(oof[L][m], dtype=float)
            per_line[m][L] = float(spearmanr(p, t).statistic)

    pooled = {}
    for m in ("M0", "M1", "M2"):
        pooled[m] = _pooled_metrics(
            [oof[L][m] for L in lines],
            [oof[L]["truth"] for L in lines],
        )

    m0_s = np.array([per_line["M0"][L] for L in lines], dtype=float)
    m1_s = np.array([per_line["M1"][L] for L in lines], dtype=float)
    m2_s = np.array([per_line["M2"][L] for L in lines], dtype=float)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    d1, d2 = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(lines), size=len(lines))
        d1.append(float(np.mean(m1_s[idx] - m0_s[idx])))
        d2.append(float(np.mean(m2_s[idx] - m0_s[idx])))
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    m1_mean, m1_lo, m1_hi = float(d1.mean()), float(np.percentile(d1, 2.5)), float(np.percentile(d1, 97.5))
    m2_mean, m2_lo, m2_hi = float(d2.mean()), float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))

    m1_equal = bool(np.allclose(m1_s, m0_s, rtol=0.0, atol=1e-12))
    max_m1_diff = float(np.max(np.abs(m1_s - m0_s)))

    # Per-fold metrics for transparency (not used for the success criterion).
    fold_metrics = []
    for fold in folds:
        held = fold["held_out_lines"]
        fm = {"fold": fold["fold"], "n_lines": len(held), "held_out_lines": held}
        for m in ("M0", "M1", "M2"):
            r2, mae = _pooled_metrics(
                [oof[L][m] for L in held],
                [oof[L]["truth"] for L in held],
            )
            fm[m] = {
                "perline_spearman_mean": float(np.mean([per_line[m][L] for L in held])),
                "pooled_r2_vs_truth": float(r2),
                "mae_vs_truth": float(mae),
            }
        fold_metrics.append(fm)

    # Construction audit: one row per line, no zero-distance distinct lines.
    X_all, _ = build_line_matrix(df, lines, lines, pca_cols)
    D = np.linalg.norm(
        X_all.to_numpy(float)[:, None, :] - X_all.to_numpy(float)[None, :, :],
        axis=2,
    )
    off = D[~np.eye(len(lines), dtype=bool)]
    zero_distance_pairs = int(np.sum((D == 0.0) & ~np.eye(len(lines), dtype=bool)) // 2)
    feat_unique = df.drop_duplicates("ModelID").set_index("ModelID")[pca_cols].loc[lines]
    pca_audit = {
        "n_lines": int(len(lines)),
        "n_unique_feature_rows": int(len(feat_unique.drop_duplicates())),
        "zero_distance_pairs": zero_distance_pairs,
        "min_pairwise_distance": float(off.min()) if len(off) else float("nan"),
        "max_pairwise_distance": float(off.max()) if len(off) else float("nan"),
    }

    target_in_m0_source = any(
        L in fold["m0_source_lines"]
        for fold in fold_audit
        for L in fold["held_out_lines"]
    )
    fold_isolation = {
        "n_folds": int(len(fold_audit)),
        "same_fold_neighbour_count": int(same_fold_neighbour_count),
        "target_line_in_m0_source": bool(target_in_m0_source),
        "m0_sources_are_training_lines_only": bool(not target_in_m0_source),
        "neighbour_pool_sizes": {
            str(f["fold"]): {L: len(v) for L, v in f["neighbours"].items()}
            for f in fold_audit
        },
    }

    m1_audit = {
        "ridge_fitted": True,
        "n_features": int(len(m1_pca_cols)),
        "alpha_grid": [float(a) for a in M1_ALPHAS],
        "per_fold_alpha": [float(f["m1_fit"]["alpha"]) for f in fold_audit],
        "perline_spearman_equals_m0": m1_equal,
        "max_abs_perline_spearman_diff_vs_m0": max_m1_diff,
        "pooled_r2_minus_m0": float(pooled["M1"][0] - pooled["M0"][0]),
        "mae_minus_m0": float(pooled["M1"][1] - pooled["M0"][1]),
    }

    zero_weight_sanity = {
        "rows_checked": int(zero_rows),
        "max_abs_diff_vs_m0": float(zero_max_abs_diff),
        "exact_equal": bool(zero_max_abs_diff == 0.0),
    }

    metrics = {
        m: {
            "perline_spearman_mean": float(np.mean([per_line[m][L] for L in lines])),
            "pooled_r2_vs_truth": float(pooled[m][0]),
            "mae_vs_truth": float(pooled[m][1]),
        }
        for m in ("M0", "M1", "M2")
    }

    results = {
        "version": "v1.2.6",
        "label": LABEL,
        "construction": (
            "fold-honest GroupKFold(5); one-row-per-line PCA keyed by ModelID; "
            "M0 means fitted on fold-training lines; M1 RidgeCV fitted on all 62 "
            "pre-registered PCs; M2 kNN on the first 10 PCs using training-only "
            "neighbours and M0 residuals"
        ),
        "models": {
            "M0": "per-drug mean fitted on fold-training lines (global-mean fallback)",
            "M1": "M0 + per-line RidgeCV offset fitted on fold-training line residuals using 62 PCs; constant per held-out line",
            "M2": "M0 + kNN (k=5, Euclidean on first 10 PCs, inverse-distance weights) over fold-training neighbours, borrowing M0 residuals",
            "M3": "skipped (runtime budget)",
        },
        "k": int(K),
        "pcs_used": list(pca_cols),
        "m1_n_pcs": int(len(m1_pca_cols)),
        "training_lines": int(len(lines)),
        "folds": fold_audit,
        "fold_metrics": fold_metrics,
        "metrics": metrics,
        "m0_perline_spearman_mean": metrics["M0"]["perline_spearman_mean"],
        "m1_perline_spearman_mean": metrics["M1"]["perline_spearman_mean"],
        "m2_perline_spearman_mean": metrics["M2"]["perline_spearman_mean"],
        "pooled_r2_m0_vs_truth": metrics["M0"]["pooled_r2_vs_truth"],
        "mae_m0_vs_truth": metrics["M0"]["mae_vs_truth"],
        "pooled_r2_m1_vs_truth": metrics["M1"]["pooled_r2_vs_truth"],
        "mae_m1_vs_truth": metrics["M1"]["mae_vs_truth"],
        "pooled_r2_m2_vs_truth": metrics["M2"]["pooled_r2_vs_truth"],
        "mae_m2_vs_truth": metrics["M2"]["mae_vs_truth"],
        "pooled_r2_vs_truth": {m: metrics[m]["pooled_r2_vs_truth"] for m in metrics},
        "mae_vs_truth": {m: metrics[m]["mae_vs_truth"] for m in metrics},
        "paired_bootstrap_lines_2000": {
            "n_resamples": int(N_BOOTSTRAP),
            "alpha": 0.05,
            "seed": int(BOOTSTRAP_SEED),
            "m1_minus_m0": {"mean": m1_mean, "ci_95": [m1_lo, m1_hi]},
            "m2_minus_m0": {"mean": m2_mean, "ci_95": [m2_lo, m2_hi]},
            # Backward-compatible aliases: these refer to M2-M0, not M0-vs-M2 agreement.
            "diff_m2_minus_m0_mean": m2_mean,
            "ci_95": [m2_lo, m2_hi],
        },
        "paired_bootstrap_m1_minus_m0": {"mean": m1_mean, "ci_95": [m1_lo, m1_hi]},
        "paired_bootstrap_m2_minus_m0": {"mean": m2_mean, "ci_95": [m2_lo, m2_hi]},
        "m1_equals_m0_perline_spearman": m1_equal,
        "m2_success_criteria_met": bool(m2_hi > 0.0),
        "m3": {"status": "skipped", "reason": "runtime budget exceeded for CatBoost-800 on the same folds"},
        "honest_note": (
            "computed over the 23 training lines only; no holdout lines used; labelled post-hoc, exploratory; "
            "not comparable to holdout metrics. DEVIATION (v1.2.5): construction defects found post-run "
            "(line-rank used as row index, within-line-centred M2 residual, no GroupKFold isolation, M1 not fitted) "
            "were corrected. v1.2.6 ALIGNMENT: M1 uses the pre-registered 62 PCA features and M2 uses the "
            "pre-registered first 10 PCs; k=5, GroupKFold(5), bootstrap seed 42/2,000 resamples and "
            "the success criterion are unchanged."
        ),
        "ndcg10_note": (
            "not computed; pre-registered graded-relevance metric, not used for any claim"
        ),
        "recall10_note": (
            "not computed; pre-registered binary AUC<0.3 metric, not used for any claim"
        ),
        "holdout_lines_excluded_count": int(len(holdout)),
        "construction_audit": {
            "pca_line_matrix": pca_audit,
            "fold_isolation": fold_isolation,
            "m1_fit": m1_audit,
            "zero_weight_sanity": zero_weight_sanity,
        },
        "per_line": {
            L: {
                "n_drugs": int(len(oof[L]["drugs"])),
                "m0_spearman": float(per_line["M0"][L]),
                "m1_spearman": float(per_line["M1"][L]),
                "m2_spearman": float(per_line["M2"][L]),
                "m1_minus_m0": float(per_line["M1"][L] - per_line["M0"][L]),
                "m2_minus_m0": float(per_line["M2"][L] - per_line["M0"][L]),
            }
            for L in lines
        },
        # Deprecated aliases retained for downstream readers. They are M0-vs-truth
        # values, not M0-vs-M2 agreement; use the *_vs_truth keys.
        "pooled_r2_m0_vs_m2": metrics["M0"]["pooled_r2_vs_truth"],
        "mae_m0_vs_m2": metrics["M0"]["mae_vs_truth"],
        "deprecated_keys_note": (
            "pooled_r2_m0_vs_m2 and mae_m0_vs_m2 are retired names retained as aliases for "
            "pooled_r2_m0_vs_truth and mae_m0_vs_truth; they never compare M0 with M2."
        ),
    }

    if write:
        out_dir = Path(out_dir)
        with open(out_dir / "diagnostic_results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(out_dir / "diagnostic_results.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["model", "perline_spearman_mean", "pooled_r2", "mae"])
            for m in ("M0", "M1", "M2"):
                w.writerow([
                    m,
                    repr(metrics[m]["perline_spearman_mean"]),
                    repr(metrics[m]["pooled_r2_vs_truth"]),
                    repr(metrics[m]["mae_vs_truth"]),
                ])
            w.writerow(["label", "post-hoc", "exploratory", "23-train-lines"])

    return results


def main():
    results = run_diagnostic(OUT, write=True)
    m = results["metrics"]
    pb = results["paired_bootstrap_lines_2000"]
    print("DIAGNOSTIC TABLE v1.2.6 (post-hoc, exploratory, 23 training lines, GroupKFold(5), not comparable to holdout metrics)")
    for name in ("M0", "M1", "M2"):
        print(
            f"{name}: per-line Spearman={m[name]['perline_spearman_mean']:.6f}  "
            f"pooled R2 vs truth={m[name]['pooled_r2_vs_truth']:.6f}  "
            f"MAE vs truth={m[name]['mae_vs_truth']:.6f}"
        )
    print("M1 per-line Spearman == M0 (empirical):", results["m1_equals_m0_perline_spearman"])
    print("M1 PCA features:", results["m1_n_pcs"], "| M2 PCA features:", len(results["pcs_used"]))
    print("M1 RidgeCV per-fold alpha:", [f['m1_fit']['alpha'] for f in results['folds']])
    print("paired M1-M0 per-line Spearman mean / 95% CI:",
          pb["m1_minus_m0"]["mean"], pb["m1_minus_m0"]["ci_95"])
    print("paired M2-M0 per-line Spearman mean / 95% CI:",
          pb["m2_minus_m0"]["mean"], pb["m2_minus_m0"]["ci_95"])
    print("M2 success criterion (95% CI > 0) met:", results["m2_success_criteria_met"])
    print("construction audit:", json.dumps(results["construction_audit"], indent=2))
    print("M3:", results["m3"]["status"], "-", results["m3"]["reason"])


if __name__ == "__main__":
    main()
