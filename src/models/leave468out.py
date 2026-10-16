"""Sensitivity check: does MDA-MB-231/MCF7/MDA-MB-468 prediction change
when MDA-MB-468 is excluded from training?

Rationale: ACH-000849 (MDA-MB-468) is in the v1 training set. If the model's
predictions for the three proposal lines are (near-)identical with and without
468 in train, the sealed prediction does not depend on having seen 468.
If they differ a lot, 468 predictions must be flagged low-confidence.

Same CatBoost config as notebook 02 §4e (800 iterations, depth 6, lr 0.05),
same seed. Compares against the frozen outputs_local/catboost_tnbc_v1.cbm.
"""
import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr, pearsonr

ROOT = '/home/abdallah/Abdallah/Projects/TNBC'
OUT = os.path.join(ROOT, 'outputs_local')
LINES = {'ACH-000768': 'MDA-MB-231', 'ACH-000849': 'MDA-MB-468',
         'ACH-000019': 'MCF7'}

ml = pd.read_parquet(os.path.join(OUT, 'tnbc_ml_dataset.parquet'))
FEAT = [c for c in ml.columns if c.startswith('pca_')
        or (c.startswith('drug_') and c not in ('drug_id', 'drug_name'))]
hold = set(np.load(os.path.join(OUT, 'holdout_lines.npy'),
                   allow_pickle=True).tolist())
hold = {h for h in hold if h in set(ml['ModelID'].unique())}

base = ml[~ml['ModelID'].isin(hold)].reset_index(drop=True)
notrain = base[base['ModelID'] != 'ACH-000849'].reset_index(drop=True)
print(f'train rows: {len(base)} | without 468: {len(notrain)} '
      f'(dropped {(base.ModelID == "ACH-000849").sum()})')

X = notrain[FEAT].to_numpy(dtype=np.float32)
y = notrain['auc'].to_numpy(dtype=np.float32)
m = CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05,
                      l2_leaf_reg=3.0, loss_function='RMSE', random_seed=42,
                      verbose=False, task_type='CPU')
m.fit(X, y)
print('retrained without 468: trees =', m.tree_count_)

orig = CatBoostRegressor()
orig.load_model(os.path.join(OUT, 'catboost_tnbc_v1.cbm'))

rows = []
for lid, name in LINES.items():
    sub = ml[ml['ModelID'] == lid]
    if len(sub) == 0:
        print(name, 'has no rows — skipped'); continue
    Xs = sub[FEAT].to_numpy(dtype=np.float32)
    p_old = np.clip(orig.predict(Xs), 0, 1)
    p_new = np.clip(m.predict(Xs), 0, 1)
    r_p = pearsonr(p_old, p_new).statistic
    r_s = spearmanr(p_old, p_new).statistic
    print(f'{name:12s} n={len(sub):5d} | Pearson(old,new)={r_p:.4f} | '
          f'Spearman(old,new)={r_s:.4f} | mean|Δ|={np.abs(p_old - p_new).mean():.4f}')
    d = sub[['drug_id']].copy()
    d['line'] = name
    d['pred_with468'] = p_old
    d['pred_without468'] = p_new
    rows.append(d)

pd.concat(rows).to_csv(os.path.join(OUT, 'leave468out_check.csv'), index=False)
print('saved outputs_local/leave468out_check.csv')
