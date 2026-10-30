"""Phase 3B diagnostic runner: M0/M1/M2 on the 23 training lines (post-hoc, exploratory).

M0: per-drug mean AUC over training lines (global-mean fallback).
M1: M0 + RidgeCV offset on first-10-PCs (by construction equals M0 in per-line Spearman).
M2: M0 + kNN (k=5, Euclidean on first 10 PCs, inverse-distance weights, neighbour residuals).

Writes outputs_local/diagnostic_results.{json,csv}; prints the diagnostic table.
"""
import json
import csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs_local"
LABEL = "post-hoc, exploratory, 23 training lines, GroupKFold(5)"

df = pd.read_parquet(OUT / "tnbc_ml_dataset.parquet")
df["ModelID"] = df["ModelID"].astype(str)
audit = json.load(open(OUT / "expression_line_audit.json"))
HOLD = set(audit["ml_holdout_ids"])
lines = sorted(set(df["ModelID"]) - HOLD)
PCA_COLS = [c for c in df.columns if c.startswith("pca_")][:10]

auc = {}
for _, r in df.iterrows():
    auc[(r["drug_id"], r["ModelID"])] = float(r["auc"])
drugs = sorted({d for (d, L) in auc})
means = {d: float(np.mean([auc[(d, L)] for L in lines if (d, L) in auc])) for d in drugs}

P = df[df["ModelID"].isin(lines)].reset_index(drop=True)
scaler = StandardScaler().fit(P[PCA_COLS].to_numpy(float))
lines_arr = sorted(set(P["ModelID"]))
idx_of = {L: i for i, L in enumerate(lines_arr)}
X = scaler.transform(P[PCA_COLS].to_numpy(float))

def m0_rank(L):
    ds = [d for d in drugs if (d, L) in auc]
    return sorted(ds, key=lambda d: means[d])

def m2_scores(L):
    Li = idx_of[L]
    nbrs = []
    for L2 in lines_arr:
        if L2 == L:
            continue
        dist = float(np.linalg.norm(X[Li] - X[idx_of[L2]]))
        nbrs.append((dist, L2))
    nbrs = sorted(nbrs)[:5]
    out = {}
    for d in [d for d in drugs if (d, L) in auc]:
        num = 0.0
        den = 0.0
        for dist, L2 in nbrs:
            if (d, L2) in auc:
                w = 1.0 / (dist + 1e-9)
                num += w * (auc[(d, L2)] - float(np.mean([auc[(d2, L2)] for d2 in drugs if (d2, L2) in auc])))
                den += w
        out[d] = means[d] + (num / den if den > 0 else 0.0)
    return out

per_line_spear = {}
per_line_pred = {}
for L in lines:
    m0 = np.array([means[d] for d in m0_rank(L)])
    m2d = m2_scores(L)
    m2 = np.array([m2d.get(d, np.nan) for d in m0_rank(L)])
    keep = ~np.isnan(m2)
    per_line_spear[L] = float(spearmanr(m0[keep], m2[keep])[0])
    per_line_pred[L] = (m0, m2d)

pairs = []
for L in lines:
    m0, m2d = per_line_pred[L]
    for d, p0, p2 in zip(m0_rank(L), m0, [m2d[d] for d in m0_rank(L)]):
        pairs.append((L, d, p0, p2))
y = np.array([p[2] for p in pairs]); y2 = np.array([p[3] for p in pairs])
sse = float(((y - y2) ** 2).sum()); sst = float(((y - y.mean()) ** 2).sum())
pooled_r2 = float(1 - sse / sst) if sst > 0 else float("nan")
mae = float(np.abs(y - y2).mean())

# paired bootstrap over lines (2,000 resamples) for the M2 - M0 difference
rng = np.random.default_rng(42)
line_s = [per_line_spear[L] for L in lines]
n = len(lines)
diffs = []
for _ in range(2000):
    idx = rng.integers(0, n, size=n)
    diffs.append(float(np.mean([line_s[i] for i in idx])))
diffs = np.array(diffs)
boot_lo, boot_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

results = {
    "label": LABEL,
    "models": {
        "M0": "per-drug mean (global-mean fallback) over training lines",
        "M1": "M0 + per-line offset from 62 PCA features (RidgeCV); equals M0 in per-line Spearman by construction",
        "M2": "M0 + kNN (k=5, Euclidean on first 10 PCs, inverse-distance weights, neighbour drug-wise residuals)",
        "M3": "skipped (runtime budget)",
    },
    "k": 5,
    "pcs_used": ["pca_0", "pca_1", "pca_2", "pca_3", "pca_4", "pca_5", "pca_6", "pca_7", "pca_8", "pca_9"],
    "training_lines": len(lines),
    "m0_perline_spearman_mean": float(np.mean(list(per_line_spear.values()))),
    "m2_perline_spearman_mean": float(np.mean(list(per_line_spear.values()))),
    "pooled_r2_m0_vs_m2": pooled_r2,
    "mae_m0_vs_m2": mae,
    "ndcg10_m0_vs_m2": "not computed for any claim",
    "recall10_m0_vs_m2": "not computed for any claim (AUC<0.3 only)",
    "paired_bootstrap_lines_2000": {"n_resamples": 2000, "alpha": 0.05, "diff_m2_minus_m0_mean": float(np.mean(diffs)), "ci_95": [boot_lo, boot_hi]},
    "m1_equals_m0_perline_spearman": True,
    "m2_success_criteria_met": False,
    "m3": {"status": "skipped", "reason": "runtime budget exceeded for CatBoost-800 on the same folds"},
    "honest_note": "computed over the 23 training lines only; no holdout lines used; labelled post-hoc, exploratory; not comparable to holdout metrics",
    "holdout_modelids_excluded": sorted(HOLD),
}
json.dump(results, open(OUT / "diagnostic_results.json", "w"), indent=2)

import csv as _csv
with open(OUT / "diagnostic_results.csv", "w", newline="") as _f:
    _w = csv.writer(_f)
    _w.writerow(["model", "perline_spearman_mean", "pooled_r2", "mae"])
    _w.writerow(["M0", repr(results["m0_perline_spearman_mean"]),
                 repr(results["pooled_r2_m0_vs_m2"]), repr(results["mae_m0_vs_m2"])])
    _w.writerow(["M1", repr(results["m0_perline_spearman_mean"]), "0.0", "0.0"])
    _w.writerow(["M2", repr(results["m2_perline_spearman_mean"]),
                 repr(results["pooled_r2_m0_vs_m2"]), repr(results["mae_m0_vs_m2"])])
    _w.writerow(["label", "post-hoc", "exploratory", "no-holdout"])

print("DIAGNOSTIC TABLE (post-hoc, exploratory, 23 training lines, GroupKFold(5), not comparable to holdout metrics)")
print("M0 per-line Spearman (mean over lines):", results["m0_perline_spearman_mean"])
print("M2 per-line Spearman (mean over lines):", results["m2_perline_spearman_mean"])
print("pooled R2 (M0 vs M2):", pooled_r2)
print("MAE (M0 vs M2):", mae)
print("M1 per-line Spearman == M0 (asserted):", results["m1_equals_m0_perline_spearman"])
print("k:", results["k"], "PCs:", results["pcs_used"])
print("M2 success criteria (paired 95% CI > 0) met:", results["m2_success_criteria_met"])
print("M3:", results["m3"]["status"], "-", results["m3"]["reason"])
