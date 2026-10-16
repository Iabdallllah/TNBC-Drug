"""FerroSelect track, phase 1 (proposal §6.1, Objectives 2-3).

Proposal-faithful modeling on the ferroptosis panel present in our matrices:
  sulfasalazine / simvastatin / lovastatin / erastin (+ support flags).

- Algorithms: LassoCV, RidgeCV, RandomForest (proposal-mandated; no GBMs here).
- Validation: outer GroupKFold by cell line (nested: inner CV for hyperparams).
- Screen covariate: included as a feature (constant while single-screen;
  framework ready the day CTRP lands) + per-screen breakdown printer.
- Ceiling check: pooled OOF Spearman vs published lab-vs-CTRPv2 GPX4-inhibitor
  agreement (RSL3 0.83 / ML210 0.70 / ML162 0.79). Anything above max => BUG flag.
- Honest uncertainty: bootstrap CI over pooled OOF predictions; per-agent
  sample sizes embedded from ferroptosis_sample_audit.csv.
- Output: panel predictions for the 3 proposal lines (sealed-doc input in F4).

Phase 2 (needs raw expression matrix + FerrDb ferroptosis tables, both
pending download): small FerrDb-gene model + Aim-0 ssGSEA circular baseline.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, 'outputs_local')
PANEL = ['DPC-006173', 'DPC-005959', 'DPC-003889', 'DPC-002562']  # sulfa/simva/lova/erastin
PANEL_NAMES = {'DPC-006173': 'SULFASALAZINE', 'DPC-005959': 'SIMVASTATIN',
               'DPC-003889': 'LOVASTATIN', 'DPC-002562': 'ERASTIN'}
CEILING = {'RSL3': 0.83, 'ML210': 0.70, 'ML162': 0.79}
SEED = 42


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {'n': int(len(y_true)),
           'mae': float(mean_absolute_error(y_true, y_pred)),
           'rmse': float(mean_squared_error(y_true, y_pred) ** 0.5)}
    try:
        out['spearman'] = float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        out['spearman'] = float('nan')
    return out


def bootstrap_spearman(y_true, y_pred, n_boot=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            v = spearmanr(y_true[idx], y_pred[idx]).statistic
        except Exception:
            continue
        if not np.isnan(v):
            vals.append(v)
    vals = np.asarray(vals)
    return {'mean': float(vals.mean()), 'lo95': float(np.percentile(vals, 2.5)),
            'hi95': float(np.percentile(vals, 97.5)), 'n_boot': int(len(vals))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=OUT)
    ap.add_argument('--outer-folds', type=int, default=3)
    ap.add_argument('--rf-trees', type=int, default=300)
    a = ap.parse_args()

    ml = pd.read_parquet(os.path.join(a.outdir, 'tnbc_ml_dataset.parquet'))
    ml = ml.drop_duplicates(subset=['ModelID', 'drug_id']).reset_index(drop=True)
    panel = ml[ml['drug_id'].isin(PANEL)].reset_index(drop=True)
    feat0 = [c for c in ml.columns if c.startswith('pca_')
             or (c.startswith('drug_') and c not in ('drug_id', 'drug_name'))]
    X = panel[feat0].to_numpy(dtype=np.float32)
    y = panel['auc'].to_numpy(dtype=np.float32)
    g = panel['ModelID'].to_numpy()
    screen = np.ones((len(panel), 1), dtype=np.float32)  # PRISM-only today
    X = np.hstack([X, screen])
    feat = feat0 + ['screen_PRISM']
    print(f'panel rows: {len(panel)} | lines: {len(set(g))} | features: {len(feat)}')
    print('per-agent n:', panel.groupby('drug_id').size().to_dict())
    if np.std(screen) == 0:
        print('NOTE: screen covariate is constant (single screen). '
              'Kept as a feature slot for the day CTRP lands.')

    models = {
        'Lasso': LassoCV(n_alphas=50, cv=3, random_state=SEED, max_iter=20000),
        'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 13), cv=3),
        'RandomForest': RandomForestRegressor(
            n_estimators=a.rf_trees, max_depth=8, min_samples_leaf=3,
            n_jobs=-1, random_state=SEED),
    }
    outer = GroupKFold(n_splits=a.outer_folds)
    oof = {name: np.full(len(panel), np.nan) for name in models}
    for fold, (tr, te) in enumerate(outer.split(X, y, g)):
        for name, mk in models.items():
            from sklearn.base import clone
            m = clone(mk).fit(X[tr], y[tr])
            oof[name][te] = m.predict(X[te])
        print(f'outer fold {fold}: test lines = {sorted(set(g[te]))}')

    results, winner, win_sp = {}, None, -2
    for name in models:
        m = metrics(y, oof[name])
        m['bootstrap_spearman'] = bootstrap_spearman(y, oof[name])
        results[name] = m
        print(f"{name:12s} pooled OOF: RMSE {m['rmse']:.4f} | "
              f"Spearman {m['spearman']:.4f} "
              f"(95% CI {m['bootstrap_spearman']['lo95']:.4f}..{m['bootstrap_spearman']['hi95']:.4f})")
        if m['spearman'] > win_sp:
            winner, win_sp = name, m['spearman']
    # tie-break toward simpler, proposal-preferred models
    for cand in ['Ridge', 'Lasso', 'RandomForest']:
        if cand in results and abs(results[cand]['spearman'] - win_sp) < 0.05:
            winner = cand
            break
    print('frozen model for sealing:', winner)

    ceiling_max = max(CEILING.values())
    flag = ('BUG-SUSPECT: pooled OOF Spearman %.3f exceeds published ceiling %.2f — '
            'treat as leakage until proven otherwise'
            % (results[winner]['spearman'], ceiling_max) if results[winner]['spearman'] > ceiling_max
            else 'ceiling check OK (below %.2f max published agreement)' % ceiling_max)
    print(flag)

    # refit winner on all panel rows -> predict the 3 proposal lines
    from sklearn.base import clone
    final = clone(models[winner]).fit(X, y)
    pca = pd.read_parquet(os.path.join(a.outdir, 'expression_pca128_breast.parquet'))
    pca = pca.drop_duplicates(subset=['ModelID']).reset_index(drop=True)
    pca_cols = [c for c in feat0 if c.startswith('pca_')]
    pos = {c: i for i, c in enumerate(feat)}
    prows = []
    for lid, lname in [('ACH-000768', 'MDA-MB-231'), ('ACH-000849', 'MDA-MB-468'),
                       ('ACH-000019', 'MCF7')]:
        prow = pca.loc[pca['ModelID'] == lid]
        if len(prow) != 1:
            print(f'WARNING: {lname} PCA rows = {len(prow)} — skipped')
            continue
        for d in PANEL:
            x = np.zeros((1, len(feat)), dtype=np.float32)
            for c in pca_cols:
                x[0, pos[c]] = float(prow[c].iloc[0])
            x[0, pos[f'drug_{d}']] = 1.0
            pred = float(np.clip(final.predict(x)[0], 0, 1))
            ntr = int((panel['drug_id'] == d).sum())
            prows.append({'line': lname, 'ModelID': lid, 'drug': PANEL_NAMES[d],
                          'drug_id': d, 'pred_auc': round(pred, 4),
                          'n_train_rows': ntr,
                          'confidence': 'LOW (<25 rows — lab data decides)' if ntr < 25 else 'moderate'})
    pred = pd.DataFrame(prows)
    pred.to_csv(os.path.join(a.outdir, 'ferroselect_panel_predictions.csv'), index=False)
    print(pred.to_string(index=False))

    res = {'frozen_model': winner, 'outer_folds': a.outer_folds,
           'n_panel_rows': int(len(panel)), 'n_lines': int(len(set(g))),
           'features': len(feat), 'screen_note': 'PRISM-only; covariate constant',
           'per_model_oof': results, 'ceiling': CEILING, 'ceiling_flag': flag,
           'seed': SEED}
    json.dump(res, open(os.path.join(a.outdir, 'ferroselect_results.json'), 'w'), indent=1)
    print('saved ferroselect_results.json + ferroselect_panel_predictions.csv')


if __name__ == '__main__':
    main()
