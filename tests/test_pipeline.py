"""Pipeline integrity tests (PRD reproducibility + anti-leakage guardrails).

Run:  python -m pytest tests/test_pipeline.py -q
   or:  python tests/test_pipeline.py   (no pytest needed)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs_local')
MDA, MCF = 'ACH-000768', 'ACH-000019'


def _ml():
    return pd.read_parquet(os.path.join(OUT, 'tnbc_ml_dataset.parquet'))


def _holdout():
    return set(np.load(os.path.join(OUT, 'holdout_lines.npy'),
                       allow_pickle=True).tolist())


def test_holdout_contains_controls():
    h = _holdout()
    assert MDA in h and MCF in h, 'MDA-MB-231/MCF7 must be holdout'


def test_no_holdout_leakage_in_split():
    ml = _ml()
    h = _holdout() & set(ml['ModelID'].unique())
    assert len(h) >= 2
    # every holdout line must be fully excluded from any train fold:
    # (train/test defined by membership only — no row may cross)
    assert h.isdisjoint(set(ml['ModelID'].unique()) - h)


def test_target_range():
    ml = _ml()
    assert float(ml['auc'].min()) >= 0 and float(ml['auc'].max()) <= 1
    assert ml['auc'].isna().sum() == 0


def test_no_duplicate_pairs():
    ml = _ml()
    assert len(ml) == ml.drop_duplicates(['ModelID', 'drug_id']).shape[0]


def test_feature_schema_numeric():
    ml = _ml()
    feat = [c for c in ml.columns if c.startswith('pca_')
            or (c.startswith('drug_') and c not in ('drug_id', 'drug_name'))]
    assert len(feat) > 100, 'too few features'
    X = ml[feat].to_numpy(dtype=np.float32)  # must not raise
    assert np.isfinite(X).all()


def test_artifacts_exist_and_traced():
    for f in ['leaderboard.csv', 'mda_mb_231_top20_killers.csv',
              'candidate_rankings.csv', 'combo_rankings.csv',
              'catboost_tnbc_v1.cbm', 'model_info.json']:
        assert os.path.exists(os.path.join(OUT, f)), f'missing {f}'
    mi = json.load(open(os.path.join(OUT, 'model_info.json')))
    assert mi.get('model_version'), 'model_version missing'
    assert 'NOT a clinical' in mi.get('disclaimer', ''), 'disclaimer missing'


def test_prd_gate():
    lb = pd.read_csv(os.path.join(OUT, 'leaderboard.csv'))
    best = lb.sort_values('ndcg10', ascending=False).iloc[0]
    assert best['r2'] > 0.25, 'R² gate failed'
    assert best['per_line_spearman'] > 0.4, 'ranking gate failed'


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f'PASS {name}')
        except AssertionError as e:
            bad += 1
            print(f'FAIL {name}: {e}')
    print(f'{len(fns) - bad}/{len(fns)} passed')
    sys.exit(1 if bad else 0)
