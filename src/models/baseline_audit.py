"""Post-hoc, exploratory baseline and selectivity audit (report v1.1).

The v1.0 holdout was already seen when these checks were designed, so every
number written here is **post-hoc and exploratory**, not a pre-registered
result.  The sealed prediction and the v1.0 artifacts are not modified.

What this script computes
-------------------------
1. A per-drug training-mean baseline on exactly the same rows as CatBoost.
   Drugs without a training row fall back to the global training mean.
2. Per-line and pooled metrics, plus a paired bootstrap over the seven
   holdout lines (the effective sample size is lines, not rows).
3. Sensitive-tail coverage of the learned 227-drug subset.
4. The predicted-vs-measured selectivity delta correlation, including the
   restricted check requested before the defence (measured AUC < 0.5 in
   either MDA-MB-231 or MCF7) and the learned-227 subset.
5. The expression-line / PCA row accounting correction (76 unique lines,
   79 rows, 3 duplicate ModelIDs, 62 fit rows / 60 unique, 16 PCA holdout).

Run:  python src/models/baseline_audit.py
"""
import json
import os
import re

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'outputs_local')
SEED = 20261029
N_BOOT = 2000


def _json(path):
    with open(path) as f:
        return json.load(f)


def _ndcg_at_10(y_true_auc, y_pred_auc, k=10):
    """NDCG@10 with graded relevance = 1 - AUC (lower AUC = more relevant)."""
    y = np.asarray(y_true_auc, dtype=float)
    p = np.asarray(y_pred_auc, dtype=float)
    rel = 1.0 - y

    def dcg(values):
        values = values[:k]
        return float(np.sum((2 ** values - 1) / np.log2(np.arange(2, len(values) + 2))))

    ideal = dcg(rel[np.argsort(y)])
    if ideal == 0:
        return 0.0
    return dcg(rel[np.argsort(p)]) / ideal


def _recall_at_10(y_true_auc, y_pred_auc, k=10, threshold=0.3):
    """Recall@10 with the binary sensitivity threshold AUC < 0.3."""
    y = np.asarray(y_true_auc, dtype=float)
    p = np.asarray(y_pred_auc, dtype=float)
    sensitive = set(np.where(y < threshold)[0])
    if not sensitive:
        return None
    return len(sensitive & set(np.argsort(p)[:k])) / len(sensitive)


def _metrics(df, pred_col):
    y = df['auc'].to_numpy(float)
    p = np.clip(df[pred_col].to_numpy(float), 0, 1)
    per_line = []
    ndcg = []
    recall = []
    for _, g in df.groupby('ModelID'):
        if len(g) > 5:
            value = spearmanr(g['auc'], g[pred_col]).statistic
            if np.isfinite(value):
                per_line.append(float(value))
        ndcg.append(_ndcg_at_10(g['auc'].values, g[pred_col].values, 10))
        r = _recall_at_10(g['auc'].values, g[pred_col].values, 10, 0.3)
        if r is not None:
            recall.append(float(r))
    return {
        'n': int(len(df)),
        'mae': float(mean_absolute_error(y, p)),
        'rmse': float(mean_squared_error(y, p) ** 0.5),
        'r2': float(r2_score(y, p)),
        'pooled_spearman': float(spearmanr(y, p).statistic),
        'per_line_spearman': float(np.mean(per_line)) if per_line else float('nan'),
        'ndcg10': float(np.mean(ndcg)),
        'recall10': float(np.mean(recall)) if recall else float('nan'),
    }


def _safe_corr(x, y, method):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return {'statistic': None, 'p': None, 'note': 'undefined (constant input)'}
    if method == 'pearson':
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)
    return {'statistic': float(r), 'p': float(p), 'note': 'ok'}


def _expression_line_audit(ml_holdout):
    """Reconcile 77/79/76/62 and prove no ML holdout line entered the PCA fit."""
    pca = pd.read_parquet(os.path.join(OUT, 'expression_pca128_breast.parquet'))
    model = pd.read_csv(os.path.join(ROOT, 'Data', 'Model.csv'), low_memory=False)
    breast = set(model.loc[model.OncotreeLineage == 'Breast', 'ModelID'].astype(str))
    ids = pca['ModelID'].astype(str).tolist()
    pca_ids = set(ids)
    wanted = breast & pca_ids

    # Historical phantom-header draw, reproduced exactly (see pca_loadings_meta.json).
    fixed = {'ACH-000768', 'ACH-000019'}
    mask = np.array([v in (breast | fixed) for v in ids])
    pool = sorted((wanted - fixed) | {'0'})
    extra = set(np.random.RandomState(42).choice(
        pool, size=max(1, len(pool) // 5), replace=False).tolist())
    pca_holdout = fixed | extra
    fit_rows = mask & ~np.array([v in pca_holdout for v in ids])
    fit_ids = set(np.array(ids)[fit_rows])
    pca_holdout_ids = set(np.array(ids)[mask]) - fit_ids

    duplicates = (pca['ModelID'].astype(str)
                  .value_counts()[lambda s: s > 1].to_dict())
    return {
        'breast_lines_annotated': len(breast),
        'expression_rows': len(ids),
        'expression_unique_lines': len(pca_ids),
        'breast_lines_with_expression': len(wanted),
        'duplicate_model_ids': {str(k): int(v) for k, v in duplicates.items()},
        'pca_fit_rows': int(fit_rows.sum()),
        'pca_fit_unique_lines': len(fit_ids),
        'pca_holdout_unique_lines': len(pca_holdout_ids),
        'pca_holdout_ids': sorted(pca_holdout_ids),
        'ml_holdout_ids': sorted(ml_holdout),
        'ml_holdout_in_pca_fit': sorted(set(ml_holdout) & fit_ids),
        'ml_holdout_in_pca_holdout': sorted(set(ml_holdout) & pca_holdout_ids),
        'all_ml_holdout_excluded_from_pca_fit': set(ml_holdout).isdisjoint(fit_ids),
        'reconciles_60_plus_16_equals_76': (len(fit_ids) == 60
                                             and len(pca_holdout_ids) == 16
                                             and len(wanted) == 76),
        'v1_0_count': 77,
        'correction': ('v1.0 reported 77 lines with expression; 77 was a row/counting '
                       'artifact. The shipped table has 79 rows and 76 unique ModelIDs.'),
    }


def main():
    ml = pd.read_parquet(os.path.join(OUT, 'tnbc_ml_dataset.parquet'))
    holdout = set(np.load(os.path.join(OUT, 'holdout_lines.npy'),
                          allow_pickle=True).tolist())
    features = [c for c in ml.columns
                if c.startswith('pca_') or (c.startswith('drug_')
                                            and c not in ('drug_id', 'drug_name'))]
    model = CatBoostRegressor()
    model.load_model(os.path.join(OUT, 'catboost_tnbc_v1.cbm'))

    train = ml[~ml['ModelID'].isin(holdout)].copy()
    test = ml[ml['ModelID'].isin(holdout)].copy()
    test['catboost'] = np.clip(
        model.predict(test[features].to_numpy(np.float32)), 0, 1)

    # Per-drug mean learned on the 23 training lines only. Unseen drugs ->
    # global training mean (the documented fallback).
    global_mean = float(train['auc'].mean())
    drug_mean = train.groupby('drug_id')['auc'].mean()
    test['per_drug_mean'] = (test['drug_id'].map(drug_mean)
                             .fillna(global_mean).clip(0, 1))
    fallback = test['drug_id'].map(drug_mean).isna()

    importance = pd.Series(model.get_feature_importance(), index=features)
    learned = {c[len('drug_'):] for c, v in importance.items()
               if c.startswith('drug_') and v > 0}
    test['learned'] = test['drug_id'].isin(learned)

    metrics = {
        'catboost': _metrics(test, 'catboost'),
        'per_drug_mean': _metrics(test, 'per_drug_mean'),
    }

    # Paired line bootstrap: resample the seven lines, not the rows.
    rng = np.random.RandomState(SEED)
    lines = sorted(test['ModelID'].unique())
    arrays = {}
    for mid, g in test.groupby('ModelID'):
        arrays[mid] = (g['auc'].to_numpy(float),
                       g['catboost'].to_numpy(float),
                       g['per_drug_mean'].to_numpy(float))
    diffs = {'per_line_spearman': [], 'r2': [], 'ndcg10': []}
    baseline_boot = {'r2': [], 'pooled_spearman': [], 'per_line_spearman': [],
                     'ndcg10': []}
    catboost_boot = {'r2': [], 'pooled_spearman': [], 'per_line_spearman': [],
                     'ndcg10': []}
    cb_better = {'per_line_spearman': 0, 'r2': 0, 'ndcg10': 0}
    for _ in range(N_BOOT):
        pick = rng.choice(lines, size=len(lines), replace=True)
        y = np.concatenate([arrays[m][0] for m in pick])
        pc = np.concatenate([arrays[m][1] for m in pick])
        pb = np.concatenate([arrays[m][2] for m in pick])
        cb_sp = np.mean([spearmanr(arrays[m][0], arrays[m][1]).statistic
                         for m in pick])
        bm_sp = np.mean([spearmanr(arrays[m][0], arrays[m][2]).statistic
                         for m in pick])
        cb_nd = np.mean([_ndcg_at_10(*arrays[m][:2]) for m in pick])
        bm_nd = np.mean([_ndcg_at_10(arrays[m][0], arrays[m][2]) for m in pick])
        vals = {
            'per_line_spearman': float(cb_sp - bm_sp),
            'r2': float(r2_score(y, pc) - r2_score(y, pb)),
            'ndcg10': float(cb_nd - bm_nd),
        }
        for k, v in vals.items():
            diffs[k].append(v)
            if v > 0:
                cb_better[k] += 1
        baseline_boot['r2'].append(float(r2_score(y, pb)))
        baseline_boot['pooled_spearman'].append(float(spearmanr(y, pb).statistic))
        baseline_boot['per_line_spearman'].append(float(bm_sp))
        baseline_boot['ndcg10'].append(float(bm_nd))
        catboost_boot['r2'].append(float(r2_score(y, pc)))
        catboost_boot['pooled_spearman'].append(float(spearmanr(y, pc).statistic))
        catboost_boot['per_line_spearman'].append(float(cb_sp))
        catboost_boot['ndcg10'].append(float(cb_nd))

    def _ci(values):
        lo, hi = np.percentile(values, [2.5, 97.5])
        return {'mean': float(np.mean(values)), 'lo95': float(lo), 'hi95': float(hi)}

    paired = {}
    for k, values in diffs.items():
        entry = _ci(values)
        entry['catboost_better_fraction'] = cb_better[k] / N_BOOT
        entry['baseline_better_fraction'] = 1 - entry['catboost_better_fraction']
        paired[k] = entry
    baseline_ci = {k: _ci(v) for k, v in baseline_boot.items()}
    catboost_ci = {k: _ci(v) for k, v in catboost_boot.items()}

    # Subsets requested before the defence.
    subsets = {
        'all_holdout': test,
        'learned_227': test[test['learned']],
        'measured_auc_below_0_5': test[test['auc'] < 0.5],
        'learned_and_measured_auc_below_0_5': test[test['learned'] & (test['auc'] < 0.5)],
        'measured_auc_below_0_3': test[test['auc'] < 0.3],
    }
    subset_metrics = {
        name: {'catboost': _metrics(d, 'catboost'),
               'per_drug_mean': _metrics(d, 'per_drug_mean')}
        for name, d in subsets.items()
    }

    calibration = {}
    for name, d in [('all_holdout', test), ('learned_227', test[test['learned']])]:
        calibration[name] = {
            'n': int(len(d)),
            'measured_mean_auc': float(d['auc'].mean()),
            'catboost_mean_auc': float(d['catboost'].mean()),
            'per_drug_mean_auc': float(d['per_drug_mean'].mean()),
            'catboost_bias': float(d['catboost'].mean() - d['auc'].mean()),
            'per_drug_mean_bias': float(d['per_drug_mean'].mean() - d['auc'].mean()),
        }

    sensitive = test[test['auc'] < 0.3]
    plateau_drugs = sorted(set(test['drug_id']) - learned)
    plateau_values = sorted({round(float(v), 6)
                             for v in test.loc[~test['learned'], 'catboost'].unique()})
    plateau_mda = sorted({round(float(v), 6) for v in test.loc[
        (~test['learned']) & (test['ModelID'] == 'ACH-000768'), 'catboost'].unique()})
    coverage = {
        'learned_drugs': len(learned),
        'plateau_drugs': len(plateau_drugs),
        'plateau_fraction': len(plateau_drugs) / max(1, test['drug_id'].nunique()),
        'plateau_prediction_values': plateau_values,
        'plateau_mda_values': plateau_mda,
        'plateau_mda_value': plateau_mda[0] if len(plateau_mda) == 1 else None,
        'holdout_rows_in_learned': int(test['learned'].sum()),
        'holdout_rows_total': int(len(test)),
        'holdout_fraction_in_learned': float(test['learned'].mean()),
        'measured_sensitive_rows': int(len(sensitive)),
        'measured_sensitive_in_learned': int(sensitive['learned'].sum()),
        'measured_sensitive_fraction_in_learned': float(sensitive['learned'].mean()),
        'per_line_top10_measured_in_learned': {
            mid: int(g.sort_values('auc').head(10)['learned'].sum())
            for mid, g in test.groupby('ModelID')
        },
        'interpretation': ('coverage only: the plateau does not hide the measured '
                           'actives; the baseline covers them trivially'),
    }

    # Selectivity delta null, including the restricted check.
    mda = test[test['ModelID'] == 'ACH-000768'][
        ['drug_id', 'auc', 'catboost']].rename(
        columns={'auc': 'auc_mda', 'catboost': 'pred_mda'})
    mcf = test[test['ModelID'] == 'ACH-000019'][
        ['drug_id', 'auc', 'catboost']].rename(
        columns={'auc': 'auc_mcf', 'catboost': 'pred_mcf'})
    delta = mda.merge(mcf, on='drug_id', how='inner')
    delta['measured_delta'] = delta['auc_mcf'] - delta['auc_mda']
    delta['predicted_delta'] = delta['pred_mcf'] - delta['pred_mda']
    delta['learned'] = delta['drug_id'].isin(learned)

    def _delta_block(d):
        return {
            'n': int(len(d)),
            'pearson': _safe_corr(d['predicted_delta'], d['measured_delta'], 'pearson'),
            'spearman': _safe_corr(d['predicted_delta'], d['measured_delta'], 'spearman'),
            'sign_agreement': float((np.sign(d['predicted_delta'])
                                     == np.sign(d['measured_delta'])).mean()),
            'predicted_delta_min': float(d['predicted_delta'].min()),
            'predicted_delta_max': float(d['predicted_delta'].max()),
            'predicted_delta_unique': int(d['predicted_delta'].nunique()),
            'measured_delta_min': float(d['measured_delta'].min()),
            'measured_delta_max': float(d['measured_delta'].max()),
        }

    selectivity = {
        'definition': 'delta = AUC(MCF7) - AUC(MDA-MB-231); positive = MDA-selective',
        'all_shared_drugs': _delta_block(delta),
        'learned_227': _delta_block(delta[delta['learned']]),
        'restricted_measured_auc_below_0_5_either': _delta_block(
            delta[(delta['auc_mda'] < 0.5) | (delta['auc_mcf'] < 0.5)]),
    }

    # The four v1.0 'selective' calls, with their measured deltas.
    flagged = ['TRIPTOLIDE', 'DOLASTATIN-10', 'ROMIDEPSIN', 'EXATECAN-MESYLATE']
    ranking = pd.read_csv(os.path.join(OUT, 'mda_mb_231_full_ranking.csv'))
    names = dict(zip(ranking['drug_id'], ranking['drug_name']))
    by_name = delta.assign(drug_name=delta['drug_id'].map(names))
    selectivity['v1_0_selective_calls'] = [
        {
            'drug_name': row['drug_name'],
            'drug_id': row['drug_id'],
            'predicted_delta': float(row['predicted_delta']),
            'measured_delta': float(row['measured_delta']),
            'measured_mda_auc': float(row['auc_mda']),
            'measured_mcf7_auc': float(row['auc_mcf']),
        }
        for _, row in by_name[by_name['drug_name'].isin(flagged)].iterrows()
    ]

    expression = _expression_line_audit(holdout)

    result = {
        'report_version': '1.1',
        'analysis_status': 'post-hoc, exploratory',
        'analysis_date_utc': '2026-10-29',
        'seed': SEED,
        'bootstrap': {'unit': 'cell line', 'n_resamples': N_BOOT,
                      'n_lines': len(lines)},
        'holdout': {
            'lines': lines,
            'n_lines': len(lines),
            'n_rows': int(len(test)),
            'n_drugs': int(test['drug_id'].nunique()),
            'training_lines': int(train['ModelID'].nunique()),
            'training_rows': int(len(train)),
            'global_training_mean': global_mean,
            'fallback_drugs_in_holdout': int(test.loc[fallback, 'drug_id'].nunique()),
            'fallback_rows': int(fallback.sum()),
            'fallback_policy': ('drug mean from the 23 training lines; drugs with no '
                                'training row use the global training mean'),
            'paired_evaluation_frame': 'identical 7,336 holdout rows for both models',
        },
        'metrics': metrics,
        'paired_bootstrap_catboost_minus_baseline': paired,
        'catboost_bootstrap_ci': catboost_ci,
        'baseline_bootstrap_ci': baseline_ci,
        'per_line': [
            {
                'ModelID': mid,
                'n': int(len(g)),
                'catboost_per_line_spearman': float(spearmanr(g['auc'], g['catboost']).statistic),
                'per_drug_mean_per_line_spearman': float(spearmanr(g['auc'], g['per_drug_mean']).statistic),
                'catboost_r2': float(r2_score(g['auc'], g['catboost'])),
                'per_drug_mean_r2': float(r2_score(g['auc'], g['per_drug_mean'])),
                'catboost_ndcg10': _ndcg_at_10(g['auc'].values, g['catboost'].values),
                'per_drug_mean_ndcg10': _ndcg_at_10(g['auc'].values, g['per_drug_mean'].values),
            }
            for mid, g in test.groupby('ModelID')
        ],
        'subsets': subset_metrics,
        'calibration': calibration,
        'coverage': coverage,
        'selectivity': selectivity,
        'expression_lines': expression,
        'ndcg_definition': ('graded relevance = 1 - AUC; lower AUC = higher relevance. '
                            'Only Recall@10 uses the binary AUC < 0.3 threshold.'),
        'limitations': [
            'The holdout was already seen before this audit; all numbers are post-hoc and exploratory.',
            'The effective sample size is 7 cell lines, not 7,336 rows.',
            'A tie within the learned 227 drugs is not proof that expression carries no signal.',
            'NDCG@10 is saturated near 0.94 for both models and is a weak discriminator.',
            'The selectivity correlation uses MDA-MB-231 and MCF7 only; it is not an external validation.',
        ],
    }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'baseline_audit.json'), 'w') as f:
        json.dump(result, f, indent=2)

    table = []
    for label, key, status in [('CatBoost-800', 'catboost', 'pre-registered holdout'),
                               ('Per-drug mean baseline', 'per_drug_mean', 'post-hoc exploratory')]:
        row = {'model': label, 'analysis_status': status}
        row.update(metrics[key])
        table.append(row)
    pd.DataFrame(table).to_csv(os.path.join(OUT, 'baseline_audit.csv'), index=False)

    with open(os.path.join(OUT, 'expression_line_audit.json'), 'w') as f:
        json.dump(expression, f, indent=2)

    print('=== Baseline audit (post-hoc, exploratory) ===')
    print(pd.DataFrame(table).to_string(index=False))
    print('\nPaired bootstrap (CatBoost - per-drug mean):')
    for k, v in paired.items():
        print(f'  {k}: {v["mean"]:+.4f} [{v["lo95"]:+.4f}, {v["hi95"]:+.4f}] '
              f'baseline better {v["baseline_better_fraction"]:.1%}')
    print('\nSelectivity delta null:')
    for k in ['all_shared_drugs', 'learned_227',
              'restricted_measured_auc_below_0_5_either']:
        s = selectivity[k]
        print(f'  {k}: n={s["n"]} pearson={s["pearson"]["statistic"]} '
              f'spearman={s["spearman"]["statistic"]} sign={s["sign_agreement"]:.3f}')
    print('\nExpression lines:', {k: expression[k] for k in
          ['expression_rows', 'expression_unique_lines',
           'breast_lines_with_expression', 'pca_fit_rows',
           'pca_fit_unique_lines', 'pca_holdout_unique_lines',
           'all_ml_holdout_excluded_from_pca_fit']})
    print('\nsaved baseline_audit.json / baseline_audit.csv / expression_line_audit.json')


if __name__ == '__main__':
    main()
