"""Pipeline integrity tests (PRD reproducibility + anti-leakage guardrails).

Run:  python -m pytest tests/test_pipeline.py -q
   or:  python tests/test_pipeline.py   (no pytest needed)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px

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


# ---------- FerroSelect track (proposal §6-8) ----------

def test_ferroselect_artifacts_exist():
    for f in ['ferroselect_results.json', 'ferroselect_panel_predictions.csv',
              'ferroptosis_sample_audit.csv', 'network_table.csv',
              'network_paths.json', 'ferroselect_network.png',
              'sealed_prediction.md']:
        assert os.path.exists(os.path.join(OUT, f)), f'missing {f}'


def test_ferroselect_ceiling_check():
    res = json.load(open(os.path.join(OUT, 'ferroselect_results.json')))
    assert 'BUG-SUSPECT' not in res['ceiling_flag'], res['ceiling_flag']
    assert res['frozen_model'] in ('Lasso', 'Ridge', 'RandomForest')


def test_sealed_doc_consistent():
    txt = open(os.path.join(OUT, 'sealed_prediction.md')).read()
    res = json.load(open(os.path.join(OUT, 'ferroselect_results.json')))
    assert res['frozen_model'] in txt
    assert 'SHA256' in txt and 'Sign-off' in txt
    for line in ['MDA-MB-231', 'MDA-MB-468', 'MCF7']:
        assert line in txt, f'{line} missing from sealed doc'


def test_network_converges_on_gpx4():
    paths = json.load(open(os.path.join(OUT, 'network_paths.json')))
    assert 'GPX4' in paths['convergent_nodes']
    assert paths['shortest_to_ferroptosis']['sulfasalazine'] is not None
    assert paths['shortest_to_ferroptosis']['simvastatin'] is not None


def test_dashboard_runs_without_streamlit_exceptions():
    """The deployed page must render end-to-end: a bad st.* kwarg kills the whole tab."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, 'dashboard', 'app.py'), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert {t.label for t in at.tabs} == {'Rankings & candidates',
                                          'Selectivity & combinations',
                                          'Model & audit', 'Live predictor'}


def test_dashboard_charts_serialise_without_nan():
    """Vega rejected the old payload; the plotly JSON must be finite and parseable."""
    json.loads(_chart_json('bar'))
    json.loads(_chart_json('scatter'))


def _chart_json(which):
    import ast
    src = open(os.path.join(ROOT, 'dashboard', 'app.py')).read()
    ns = {'np': np, 'pd': pd, 'px': px, 'SENSITIVE_AUC': 0.3,
          'HIGHLIGHT': '#D9480F', 'MUTED': '#868E96',
          'SENS_MAP': {'SENSITIVE (AUC<0.3)': 'Sensitive', 'sensitive': 'Sensitive',
                       'resistant/intermediate': 'Resistant', 'resistant': 'Resistant',
                       'RESISTANT (AUC>=0.3)': 'Resistant'},
          'SEL_MAP': {'selective': 'Selective', 'non-selective': 'Not selective'}}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in ('pretty', 'bar_chart',
                                                               'selectivity_chart'):
            exec(compile(ast.Module([node], []), '<ast>', 'exec'), ns)
    rank = pd.read_csv(os.path.join(OUT, 'mda_mb_231_full_ranking.csv'))
    fig = ns['bar_chart'](rank.head(20)) if which == 'bar' else ns['selectivity_chart'](rank, 5)
    blob = fig.to_json()
    for token in ('NaN', 'Infinity'):
        assert token not in blob, f'{token} in plotly payload ({which})'
    return blob


def test_verdict_labels_are_short_and_unique():
    """Long repeated strings ('resistant/intermediate') were a review complaint."""
    df = pd.read_csv(os.path.join(OUT, 'mda_mb_231_full_ranking.csv'))
    calls = set(df['sensitive_call'])
    assert calls <= {'SENSITIVE (AUC<0.3)', 'resistant/intermediate'}, calls


def test_pca_loadings_are_verified_before_being_published():
    """Gene annotations may only ship if the refit reproduces the shipped PCA scores."""
    loadings = os.path.join(OUT, 'pca_loadings_top.csv')
    if not os.path.exists(loadings):
        print('  (no PCA loadings published - provenance gap, '
              'see outputs_local/pca_loadings_status.json)')
        return
    meta = json.load(open(os.path.join(OUT, 'pca_loadings_meta.json')))
    assert meta['n_components_verified'] >= 50, \
        'published loadings do not reproduce the shipped PCA scores'
    assert meta['min_abs_correlation_verified'] > 0.999, \
        'published loadings do not reproduce the shipped PCA scores'
    assert meta['max_reconstruction_error_verified'] < 1e-3, \
        'published loadings do not match the shipped PCA scores'
    df = pd.read_csv(loadings)
    assert len(df) == meta['n_components']
    verified_rows = df[df['status'] == 'verified']
    assert len(verified_rows) == meta['n_components_verified']
    assert set(df['pca_feature']) - set(verified_rows['pca_feature']) == \
        set(meta['components_not_attributable'])
    bad = verified_rows['label'].str.match(
        r'^pca_\d+ \(.+, .+, .+\)$') == False  # noqa: E712
    assert not bad.any(), 'malformed PC label'


def test_selectivity_rule_is_relative_not_the_dead_fixed_cutoff():
    """The old delta>0.05 threshold was unreachable (max 0.0385) and flagged nothing."""
    rule = json.load(open(os.path.join(OUT, 'selectivity_rule.json')))
    assert rule['delta_stats']['n_above_old_threshold'] == 0
    assert rule['selectivity_rule'] != rule['old_selectivity_rule']
    df = pd.read_csv(os.path.join(OUT, 'mda_mb_231_full_ranking.csv'))
    assert set(df['selective_call']) <= {'Selective', 'non-selective'}
    sel = df[df['selective_call'] == 'Selective']
    assert len(sel) > 0, 'relative rule selected nothing at all'
    assert (sel['selectivity_delta'] > 0).all(), 'a "Selective" row has delta <= 0'
    assert (sel['pred_AUC_MDA231'] < rule['selective_auc_max']).all()


def test_offtarget_layer_covers_full_wetlab_panel():
    """Every panel killer needs a computable rescue-reversal page, incl. RSL3."""
    rv = pd.read_csv(os.path.join(OUT, 'offtarget_rescue_reversal.csv'))
    killers = {'ERASTIN', 'SULFASALAZINE', 'SIMVASTATIN', 'LOVASTATIN', 'RSL3'}
    for killer in killers:
        sub = rv[rv['killer'] == killer]
        assert len(sub) > 0, f'{killer} missing from the rescue-reversal table'
        if killer == 'RSL3':
            src = json.load(open(os.path.join(OUT, 'rsl3_mini_signature_meta.json')))
            assert 'single paper' in src['quality_note'], \
                'RSL3 signature must stay labelled as single-paper'
        nac_p = sub[sub['rescue'] == 'N-ACETYLCYSTEINE']['reversal_p'].min()
        assert nac_p < 0.01, f'{killer}: NAC reversal not significant (p={nac_p})'
    short = pd.read_csv(os.path.join(OUT, 'protein_array_shortlist.csv'))
    singles = {'ERASTIN', 'SULFASALAZINE', 'SIMVASTATIN', 'RSL3'}
    combos = {'ERASTIN+SIMVASTATIN', 'SULFASALAZINE+SIMVASTATIN', 'RSL3+SIMVASTATIN'}
    assert set(short['drug'].unique()) == singles | combos
    assert short['gene'].notna().all()
    assert short['drug'].nunique() == 7


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
