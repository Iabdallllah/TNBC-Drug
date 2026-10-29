"""Calibrate prediction intervals for the v1 CatBoost AUC model.

The champion model ships point predictions only. For the dashboard (and the
defence report) we need an honest error bar per prediction. CatBoost was not
trained with a quantile objective, so tree-variance intervals are not available
from the saved artifact; instead we use a conformal-style calibration on the
seven pre-registered holdout cell lines, which the model never saw during
training or tuning.

Output: outputs_local/prediction_intervals.json

The JSON carries:
  * global residual quantiles (80% and 95% intervals),
  * per-holdout-line intervals,
  * per-drug intervals where a drug has enough holdout rows (n >= MIN_DRUG_N).
Predictions outside those groups fall back to the global interval.

Run:  python src/models/prediction_intervals.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'outputs_local')
DATASET = os.path.join(OUT, 'tnbc_ml_dataset.parquet')
MODEL = os.path.join(OUT, 'catboost_tnbc_v1.cbm')
HOLDOUT = os.path.join(OUT, 'holdout_lines.npy')
TARGET = os.path.join(OUT, 'prediction_intervals.json')
MIN_DRUG_N = 40          # below this a per-drug quantile is just noise


def feature_cols(df):
    return [c for c in df.columns if c.startswith('pca_')
            or (c.startswith('drug_') and c not in ('drug_id', 'drug_name'))]


def interval_stats(residual, prefix=''):
    """Empirical two-sided intervals + one-sided absolute-error quantiles."""
    r = np.asarray(residual, dtype=float)
    return {
        f'{prefix}n': int(len(r)),
        f'{prefix}rmse': float(np.sqrt(np.mean(r ** 2))),
        f'{prefix}mae': float(np.mean(np.abs(r))),
        f'{prefix}sigma': float(np.std(r, ddof=1)),
        f'{prefix}q025': float(np.quantile(r, 0.025)),
        f'{prefix}q975': float(np.quantile(r, 0.975)),
        f'{prefix}abs_q80': float(np.quantile(np.abs(r), 0.80)),
        f'{prefix}abs_q95': float(np.quantile(np.abs(r), 0.95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=TARGET)
    a = ap.parse_args()

    for p in (DATASET, MODEL, HOLDOUT):
        if not os.path.exists(p):
            raise SystemExit(f'missing input: {p}')

    df = pd.read_parquet(DATASET)
    feat = feature_cols(df)
    holdout = [str(x) for x in np.load(HOLDOUT, allow_pickle=True).tolist()]
    hold = df[df['ModelID'].astype(str).isin(holdout)].copy()
    if hold.empty:
        raise SystemExit('no holdout rows found in the dataset')

    model = CatBoostRegressor()
    model.load_model(MODEL)
    pred = np.clip(model.predict(hold[feat].to_numpy(dtype=np.float32)), 0, 1)
    hold['residual'] = pred - hold['auc'].to_numpy(dtype=float)

    global_stats = interval_stats(hold['residual'].to_numpy())

    lines = {}
    for mid, sub in hold.groupby('ModelID'):
        lines[str(mid)] = interval_stats(sub['residual'].to_numpy(), prefix='')

    drugs = {}
    for did, sub in hold.groupby('drug_id'):
        if len(sub) >= MIN_DRUG_N:
            s = interval_stats(sub['residual'].to_numpy(), prefix='')
            s['abs_q80'] = max(s['abs_q80'], global_stats['abs_q80'] * 0.5)
            s['abs_q95'] = max(s['abs_q95'], global_stats['abs_q95'] * 0.5)
            drugs[str(did)] = s

    payload = {
        'source': 'src/models/prediction_intervals.py',
        'method': ('conformal-style empirical intervals from residuals on the '
                   'pre-registered holdout cell lines (never seen in training)'),
        'calibration_lines': sorted(lines),
        'n_holdout_lines': len(lines),
        'n_rows': int(len(hold)),
        'min_drug_n': MIN_DRUG_N,
        'n_drugs_with_own_interval': len(drugs),
        'global': global_stats,
        'lines': lines,
        'drugs': drugs,
        'note': ('Intervals are marginal, not conditional guarantees. Use the 80% '
                 'half-width as the default error bar and the 95% half-width for '
                 'decision limits. Per-drug intervals are only emitted when the '
                 'drug has at least 40 holdout rows; otherwise the line/global '
                 'fallback applies.'),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'wrote {a.out}')
    print(f"  holdout rows={len(hold):,} lines={len(lines)} "
          f"global 80%=±{global_stats['abs_q80']:.3f} 95%=±{global_stats['abs_q95']:.3f}")
    print(f"  per-drug intervals: {len(drugs)}")


if __name__ == '__main__':
    main()
