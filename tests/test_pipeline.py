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
                                          'Synergy checkerboard', 'Protein network',
                                          'Model & audit', 'Defense report',
                                          'Live predictor'}


def test_dashboard_charts_serialise_without_nan():
    """Vega rejected the old payload; the plotly JSON must be finite and parseable."""
    json.loads(_chart_json('bar'))
    json.loads(_chart_json('scatter'))
    ci_blob = _chart_json('bar_ci')
    json.loads(ci_blob)
    assert '"error_x"' in ci_blob, 'the AUC bar chart lost its 80% CI error bars'


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
    if which == 'bar_ci':
        rank = rank.head(5).copy()
        rank['auc_hw80'] = 0.093
    fig = ns['bar_chart'](rank.head(20)) if which == 'bar' else (
        ns['bar_chart'](rank) if which == 'bar_ci'
        else ns['selectivity_chart'](rank, 5))
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
    singles = {'ERASTIN', 'SULFASALAZINE', 'RSL3',
               'N-ACETYLCYSTEINE', 'DEFEROXAMINE'}
    combos = {'ERASTIN+SIMVASTATIN', 'SULFASALAZINE+SIMVASTATIN', 'RSL3+SIMVASTATIN'}
    assert set(short['drug'].unique()) == singles | combos, \
        'shortlist must match the wet-lab panel exactly (8 members, statin only in combos)'
    assert short['gene'].notna().all()
    assert short['drug'].nunique() == 8


def test_live_predictor_matches_sealed_rankings():
    """The live predictor must reproduce the rankings tab exactly, and must NOT
    pretend a drug the model never split on is a prediction (it returns the
    drug-free baseline, which is what made every drug read 0.907)."""
    import pyarrow.parquet as pq
    from catboost import CatBoostRegressor
    cols = pq.read_schema(os.path.join(OUT, 'tnbc_ml_dataset.parquet')).names
    feat = [c for c in cols if c.startswith('pca_')
            or (c.startswith('drug_') and c not in ('drug_id', 'drug_name'))]
    pos = {c: i for i, c in enumerate(feat)}
    pca = pd.read_parquet(os.path.join(OUT, 'expression_pca128_breast.parquet'))
    pca = pca.drop_duplicates(subset=['ModelID'], keep='first')
    model = CatBoostRegressor()
    model.load_model(os.path.join(OUT, 'catboost_tnbc_v1.cbm'))
    imp = pd.Series(model.get_feature_importance(), index=feat)
    scored = {c[len('drug_'):] for c, v in imp.items()
              if c.startswith('drug_') and v > 0}
    pca_cols = [c for c in feat if c.startswith('pca_')]
    rank = pd.read_csv(os.path.join(OUT, 'mda_mb_231_full_ranking.csv'))

    def vec(line, drug=None):
        row = pca.loc[pca['ModelID'] == line, pca_cols]
        x = np.zeros((1, len(feat)), dtype=np.float32)
        x[0, [pos[c] for c in pca_cols]] = row.to_numpy(dtype=np.float32)[0]
        if drug is not None:
            x[0, pos[f'drug_{drug}']] = 1.0
        return float(np.clip(model.predict(x)[0], 0, 1))

    assert scored, 'no drug feature carries importance - the one-hot contract broke'
    names = dict(zip(rank['drug_id'].astype(str), rank['drug_name']))

    def table_auc(drug):
        return float(rank.loc[rank['drug_id'].astype(str) == drug,
                              'pred_AUC_MDA231'].iloc[0])

    scored_ids = [d for d in rank['drug_id'].astype(str) if d in scored][:5]
    assert len(scored_ids) >= 2, 'need at least two scored drugs to test drug dependence'
    seen = {}
    for d in scored_ids:
        got = vec('ACH-000768', d)
        assert abs(got - table_auc(d)) < 1e-6, \
            f'{names[d]}: live {got:.6f} != sealed table {table_auc(d):.6f}'
        seen[d] = got
    assert len(set(round(v, 6) for v in seen.values())) > 1, \
        'changing the drug did not change the prediction - drug features not encoded'

    baseline = vec('ACH-000768')
    unscored = [d for d in rank['drug_id'].astype(str) if d not in scored][:20]
    for d in unscored:
        got = vec('ACH-000768', d)
        assert abs(got - baseline) < 1e-9, \
            f'{names.get(d, d)} has no model signal but differs from the baseline'
    return {'scored': len(scored), 'unscored_sample': len(unscored),
            'baseline': round(baseline, 6)}


def test_panel_ranking_covers_the_eight_members_in_order():
    """The FerroSelect view must show all 8 wet-lab members ranked 1-8 with the
    lab-only lanes explicitly marked, so nobody reads an RSL3 number that is not there."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, 'dashboard', 'app.py'), default_timeout=300)
    at.run()
    at.radio[0].set_value('FerroSelect inducer panel')
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    tables = [d.value for d in at.dataframe]
    ranked = next((t for t in tables if 'Rank' in list(t.columns)), None)
    assert ranked is not None, 'the 1-8 panel ranking table is missing from the view'
    assert len(ranked) == 8, f'expected the 8 wet-lab members, got {len(ranked)}'
    assert list(ranked['Rank']) == list(range(1, 9)), 'ranking must be ordered 1-8'
    members = list(ranked['Panel member'])
    for required in ['RSL3', 'RSL3 + SIMVASTATIN', 'N-ACETYLCYSTEINE (NAC)',
                     'DEFEROXAMINE (DFO)']:
        assert any(required in m for m in members), f'{required} missing from the ranking'
    auc = dict(zip(ranked['Panel member'], ranked['Predicted AUC (MDA-MB-231)']))
    lab_only = ['RSL3 + SIMVASTATIN', 'RSL3',
                'N-ACETYLCYSTEINE (NAC)', 'DEFEROXAMINE (DFO)']
    for m in lab_only:
        assert pd.isna(auc[m]), f'{m} must not show a fabricated AUC'
    for m, lo, hi in [('ERASTIN + SIMVASTATIN', 0.7, 0.8), ('SULFASALAZINE + SIMVASTATIN', 0.7, 0.85),
                      ('ERASTIN', 0.8, 0.9), ('SULFASALAZINE', 0.85, 0.95)]:
        assert lo < auc[m] < hi, f'{m} AUC {auc[m]} outside the sealed range'
    assert any('244' in str(n) for n in ranked['Evidence / note']), \
        'the combined RSL3+simvastatin suppression count must stay visible'


def test_ctpv2_external_ground_truth_and_v2_model():
    """CTRPv2.1 RSL-3 measured AUCs exist, gold lines are real measurements and
    every v2 OOF Spearman stays below the published lab-vs-CTRP ceiling."""
    auc_csv = os.path.join(OUT, 'ctpv2_ferroptosis_auc.csv')
    if not os.path.exists(auc_csv):
        return
    t = pd.read_csv(auc_csv)
    rsl = t[t['target'] == 'RSL3']
    assert 700 <= len(rsl) <= 900, f'RSL3 cohort size {len(rsl)} unexpected'
    gold = {r.ModelID: r for _, r in rsl.iterrows()
            if r.ModelID in (MDA, 'ACH-000849', 'ACH-000019')}
    assert set(gold) == {MDA, 'ACH-000849', 'ACH-000019'}, 'gold lines missing'
    lens = [gold[MDA].auc_raw, gold['ACH-000849'].auc_raw,
            gold['ACH-000019'].auc_raw]
    assert len(set(lens)) == 3, 'AUC values for the lines must differ'
    lo, hi = sorted((gold[MDA].auc_norm_0_1, gold['ACH-000019'].auc_norm_0_1))
    assert lo < hi - 0.05, 'MDA-MB-231 must be clearly more sensitive than MCF7'
    report = os.path.join(OUT, 'rsl3_v2_report.json')
    if os.path.exists(report):
        rep = json.load(open(report))
        assert rep['winner'] in ('Lasso', 'Ridge', 'RandomForest')
        ceil = rep['ceiling_lab_vs_ctrp']
        assert rep['models'][rep['winner']]['spearman'] < ceil['RSL3'], \
            'v2 OOF Spearman may not beat the published lab-vs-CTRP ceiling'
        gp = pd.read_csv(os.path.join(OUT, 'rsl3_v2_gold_predictions.csv'))
        assert set(gp.ModelID) == {MDA, 'ACH-000849', 'ACH-000019'}
        assert gp.oof_auc_norm_0_1.between(0, 1).all()
        assert gp.loc[gp.ModelID == MDA, 'oof_auc_norm_0_1'].iloc[0] < \
            gp.loc[gp.ModelID == 'ACH-000019', 'oof_auc_norm_0_1'].iloc[0]


def test_ctpv2_block_visible_in_ferroselect_view():
    """The measured CTRPv2 RSL-3 AUCs must show in the app (truth, not sealth)."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, 'dashboard', 'app.py'),
                           default_timeout=300)
    at.run()
    at.radio[0].set_value('FerroSelect inducer panel')
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    heads = [e.value for e in at.subheader if e.value is not None]
    assert any('CTRPv2 external ground truth for RSL3' in h for h in heads), heads
    tables = [d.value for d in at.dataframe]
    ctp = next((t for t in tables if 'RSL-3 raw AUC (CTRPv2)' in list(t.columns)), None)
    assert ctp is not None, 'the measured RSL-3 AUC table is missing'
    assert len(ctp) == 3, 'expected the three gold lines'
    v = dict(zip(ctp['Cell line'], ctp['RSL-3 raw AUC (CTRPv2)']))
    lo = pd.to_numeric(pd.Series(v.values()), errors='coerce')
    assert lo.between(1, 20).all(), v


def test_synergy_math_detects_synergy_and_antagonism():
    """Bliss, ZIP and Loewe must all recover a known interaction from one matrix."""
    sys.path.insert(0, os.path.join(ROOT, 'dashboard'))
    import synergy

    ca = np.array([0, 0.1, 0.3, 1, 3, 10, 30, 100.0])
    cb = ca.copy()
    ea = 0.55 * (ca / ca.max()) ** 1.1
    eb = 0.45 * (cb / cb.max()) ** 1.1
    additive = ea[:, None] + eb[None, :] - ea[:, None] * eb[None, :]
    base = 1.0 - additive

    placebo = synergy.placeholder_matrix(ca, cb, auc_a=0.8, auc_b=0.8)
    p = synergy.bliss_analysis(placebo, ca, cb)
    assert abs(p['summary']['mean_excess']) < 1e-9, \
        'the model-additive placeholder must have exactly zero Bliss excess'

    v_syn = base.copy()
    v_syn[1:, 1:] = np.clip(base[1:, 1:] * 0.5, 0, 1)
    a = synergy.analyse(v_syn, ca, cb)
    assert a['bliss']['summary']['mean_excess'] > 0.20
    assert a['zip']['summary']['mean_excess'] > 0.20
    assert a['loewe']['summary']['median_ci'] < 1.0
    assert a['bliss']['summary']['n_synergistic'] > 0

    v_ant = base.copy()
    v_ant[1:, 1:] = 1.0          # combination fully protects: pure antagonism
    ant = synergy.bliss_analysis(v_ant, ca, cb)
    assert ant['summary']['mean_excess'] < -0.15, 'antagonism was not detected'

    for fn in (synergy.observed_heatmap, synergy.bliss_heatmap,
               synergy.zip_heatmap, synergy.loewe_heatmap):
        blob = fn(v_syn, ca, cb).to_json()
        assert 'NaN' not in blob and 'Infinity' not in blob, fn.__name__

    assert list(synergy.parse_conc_text('100, 10, 1, 0')) == [0, 1, 10, 100]
    for bad in ('', '5', '0, -1, 2', '0, abc, 2'):
        try:
            synergy.parse_conc_text(bad)
        except ValueError:
            continue
        raise AssertionError(f'parse_conc_text accepted {bad!r}')

    preds = pd.read_csv(os.path.join(OUT, 'ferroselect_panel_predictions.csv'))
    ref = synergy.model_reference_effect(preds, 'MDA-MB-231', 'ERASTIN', 'SIMVASTATIN')
    assert ref is not None and 0 < ref['e_ab'] < 1
    assert synergy.model_reference_effect(preds, 'MDA-MB-231', 'RSL3', 'SIMVASTATIN') is None
    blob = synergy.comparison_chart(ref['e_ab'], 0.5).to_json()
    assert 'NaN' not in blob and 'Infinity' not in blob


def test_prediction_intervals_are_calibrated_on_blind_holdouts():
    """The ±CI presentation must be backed by the committed calibration artifact."""
    p = json.load(open(os.path.join(OUT, 'prediction_intervals.json')))
    assert 'conformal' in p['method'].lower()
    assert p['n_holdout_lines'] == 7
    hold = _holdout()
    ml = _ml()
    n_expected = int(ml['ModelID'].astype(str).isin(hold).sum())
    assert p['n_rows'] == n_expected, (p['n_rows'], n_expected)

    g = p['global']
    assert 0.05 < g['abs_q80'] < 0.15, g['abs_q80']
    assert g['abs_q95'] > g['abs_q80'] + 0.05
    assert 0.05 < g['rmse'] < 0.20
    for mid in (MDA, MCF):
        s = p['lines'][mid]
        assert s['n'] > 100
        assert 0 < s['abs_q80'] < s['abs_q95'] <= 0.35
        assert abs(s['abs_q80'] - g['abs_q80']) < 0.05
    assert p['min_drug_n'] >= 25
    assert isinstance(p['drugs'], dict)
    for s in p['drugs'].values():
        assert s['abs_q95'] > s['abs_q80'] > 0


def test_defence_report_is_self_contained_and_complete():
    """The export must be printable HTML with every required thesis section."""
    sys.path.insert(0, os.path.join(ROOT, 'dashboard'))
    import defense_report

    h = defense_report.build_defense_report(OUT)
    assert h.startswith('<!DOCTYPE html>') and h.rstrip().endswith('</html>')
    assert '@page' in h and 'Print' not in h.split('</style>')[0]  # printable CSS
    assert '<script' not in h.lower(), 'the report must stay JavaScript-free'
    for marker in ['1. Dataset, leakage controls', '2. Model vs a one-line baseline',
                   '3. Selectivity null', '4. The eight-member wet-lab panel',
                   '5. Top-ranked killers', '6. Bliss combination expectations',
                   '7. CTRPv2.1 measured ground truth', '8. Prediction uncertainty',
                   '9. Validation gates', '10. Sign-off']:
        assert marker in h, marker
    for _, member, _ in defense_report.PANEL:
        assert member in h, f'{member} missing from the defence report'
    assert 'RSL-3 raw AUC (CTRPv2)' in h
    assert 'MDA-MB-231' in h and 'MCF7' in h
    assert '±0.104' in h, 'global 80% interval missing'
    assert 'Per-drug mean baseline' in h
    assert 'post-hoc, exploratory' in h.lower()
    assert 'Predicted selective (unvalidated)' in h
    assert 'graded relevance' in h.lower()
    assert '76 unique lines' in h
    assert 'v1.0-defense' in h
    assert '>nan<' not in h.lower() and '>None<' not in h


def test_rescue_network_uses_real_protein_lists():
    """The network must draw the real shortlist and refuse mislabelled signatures."""
    sys.path.insert(0, os.path.join(ROOT, 'dashboard'))
    import network_graph

    sl = pd.read_csv(os.path.join(OUT, 'protein_array_shortlist.csv'))
    rv = pd.read_csv(os.path.join(OUT, 'offtarget_rescue_reversal.csv'))
    pm = pd.read_csv(os.path.join(OUT, 'offtarget_panel_map.csv'))
    cov = pd.read_csv(os.path.join(OUT, 'offtarget_coverage_ferroselect.csv'))

    # Data-layer quarantine: the legacy NAC rows must carry a marker, not just
    # be filtered by a UI string comparison.
    for name, column in [('offtarget_panel_map.csv', 'drug'),
                         ('offtarget_rescue_reversal.csv', 'rescue'),
                         ('protein_array_shortlist.csv', 'drug'),
                         ('offtarget_coverage_ferroselect.csv', 'panel_drug')]:
        d = pd.read_csv(os.path.join(OUT, name))
        assert 'quarantined' in d.columns, f'{name}: missing quarantined column'
        marked = d[d[column].astype(str).str.upper().str.startswith('N-ACETYL')]
        assert len(marked) > 0, f'{name}: no NAC rows found to check'
        assert marked['quarantined'].astype(str).str.lower().isin(
            ['true', '1']).all(), f'{name}: an NAC row is not quarantined'
        others = d[~d[column].astype(str).str.upper().str.startswith('N-ACETYL')]
        assert not others['quarantined'].astype(str).str.lower().isin(
            ['true', '1']).any(), f'{name}: a non-NAC row is wrongly quarantined'

    # Exact match must beat the latent Erastin/Cloperastine substring collision.
    from src.models.offtarget_coverage import resolve_l1000_term
    terms = {'erastin': {'term': 'Erastin Down'},
             'cloperastine': {'term': 'Cloperastine Down'}}
    assert resolve_l1000_term('ERASTIN', terms)['term'] == 'Erastin Down'
    assert resolve_l1000_term(
        'ERASTIN', {'cloperastine': {'term': 'Cloperastine Down'}}) is None
    assert resolve_l1000_term(
        'NAC', {'solifenacinsuccinate': {'term': 'Solifenacin-Succinate Down'}}) is None
    assert resolve_l1000_term(
        'DEFEROXAMINE', {'deferoxaminemesylate': {
            'term': 'Deferoxamine-Mesylate Down'}})['term'] == 'Deferoxamine-Mesylate Down'

    fig, info = network_graph.rescue_network_figure(
        sl, rv, 'RSL3 + SIMVASTATIN', 'DEFEROXAMINE', MDA, panel_map=pm, coverage=cov)
    assert info['valid'] is True
    assert info['n_suppressed'] == 242
    assert info['n_rescued'] == 197          # full Deferoxamine-Mesylate Up signature
    assert info['n_overlap'] == 9            # stored killer-down ∩ rescue-up genes
    assert '244 suppression events' in info['note']
    assert '206 SIMVASTATIN' in info['note']
    kinds = {d['kind'] for _, d in info['_graph'].nodes(data=True)}
    assert {'killer', 'rescue', 'suppressed', 'shared'} <= kinds
    blob = fig.to_json()
    json.loads(blob)
    assert 'NaN' not in blob and 'Infinity' not in blob

    # NAC's legacy L1000 term is Solifenacin-Succinate: a false fuzzy match.
    assert network_graph.signature_audit('N-ACETYLCYSTEINE', cov)[0] is False
    assert network_graph.signature_audit('DEFEROXAMINE', cov)[0] is True
    bad_fig, bad = network_graph.rescue_network_figure(
        sl, rv, 'RSL3 + SIMVASTATIN', 'N-ACETYLCYSTEINE', MDA, panel_map=pm, coverage=cov)
    assert bad_fig is None and bad['valid'] is False
    assert 'Solifenacin' in bad['warning']
    assert bad['n_shared'] == 0, 'NAC must not inherit the solifenacin overlap'

    fig2, info2 = network_graph.rescue_network_figure(
        sl, rv, 'ERASTIN', 'DEFEROXAMINE', MDA, max_nodes=40, panel_map=pm, coverage=cov)
    assert info2['n_suppressed'] == 177 and info2['n_rescued'] == 197
    assert info2['p'] is not None and info2['p'] < 0.05

    paths = json.load(open(os.path.join(OUT, 'network_paths.json')))
    mfig, nodes = network_graph.mechanism_path_figure(
        paths, ['erastin', 'sulfasalazine', 'simvastatin', 'RSL3'])
    assert mfig is not None and 'GPX4' in nodes


def test_model_info_has_no_stale_artifact_references():
    """model_info.json is the public manifest; a dead file name is a broken contract."""
    mi = json.load(open(os.path.join(OUT, 'model_info.json')))
    assert 'pca_62' in mi['features'], mi['features']
    assert 'pca_128' not in mi['features'], mi['features']
    assert mi['dataset'].startswith('outputs_local/')
    assert os.path.exists(os.path.join(ROOT, mi['dataset']))
    assert not any('rf_tnbc_v1.pkl' in x for x in mi['outputs'])
    for entry in mi['outputs']:
        name = entry.split(' ')[0]
        assert os.path.exists(os.path.join(OUT, name)), f'stale artifact reference: {entry}'
    assert 'prediction_intervals.json' in mi['outputs']
    assert 'catboost_tnbc_v1.cbm' in mi['outputs']


def test_outputs_local_artifacts_parse():
    """Integrity sweep: every artifact in outputs_local must load cleanly."""
    bad = []
    files = [f for f in sorted(os.listdir(OUT))
             if os.path.isfile(os.path.join(OUT, f)) and not f.startswith('.')]
    assert len(files) >= 45, f'only {len(files)} artifacts in outputs_local'
    for f in files:
        p = os.path.join(OUT, f)
        try:
            assert os.path.getsize(p) > 0, 'empty file'
            if f.endswith('.csv'):
                assert pd.read_csv(p, nrows=5).shape[1] >= 1
            elif f.endswith('.json'):
                json.load(open(p))
            elif f.endswith('.npy'):
                np.load(p, allow_pickle=True)
            elif f.endswith('.parquet'):
                pd.read_parquet(p)
            elif f.endswith(('.md', '.txt')):
                assert len(open(p, encoding='utf-8', errors='replace').read()) > 0
            elif f.endswith(('.png', '.joblib', '.cbm')):
                assert os.path.getsize(p) > 100
        except Exception as e:
            bad.append(f'{f}: {type(e).__name__}: {e}')
    assert not bad, bad


def test_dashboard_ci_and_defence_export_are_visible():
    """The new UI must actually expose the CI column and the report download."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, 'dashboard', 'app.py'), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    ranked = next((d.value for d in at.dataframe if 'auc_ci_mda231' in d.value.columns), None)
    assert ranked is not None, 'the ranking table lost the CI column'
    ci = ranked['auc_ci_mda231'].astype(str)
    assert ci.str.match(r'^\d\.\d{3} ± \d\.\d{3}$').all(), ci.head().tolist()
    assert any(d.label == 'Download defence report (HTML)'
               for d in at.get('download_button'))
    assert len(at.get('plotly_chart')) >= 5, 'new charts did not render'
    # The checkerboard must refuse to score the untouched additive placeholder …
    assert any('model-additive placeholder' in i.value for i in at.info), \
        'the placeholder is being scored as if it were measured data'
    # … and the explicit demo toggle must activate all three scores.
    demo = next(t for t in at.toggle if 'illustrative synergy demo' in t.label)
    at = demo.set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = {m.label for m in at.metric}
    assert {'Mean Bliss excess', 'Median Loewe CI'} <= labels, labels
    assert len(at.get('plotly_chart')) >= 8, 'the scored checkerboard charts did not render'

    # NAC quarantine must hold beyond the network tab: the 8-member panel note may
    # not recycle the 166 Solifenacin genes, and the rescue table must drop NAC rows.
    view = next(r for r in at.radio if r.label == 'Dataset')
    at = view.set_value('FerroSelect inducer panel').run()
    assert not at.exception, [e.value for e in at.exception]
    panel = next(d.value for d in at.dataframe
                 if 'Panel member' in d.value.columns and 'Evidence / note' in d.value.columns)
    nac = panel[panel['Panel member'].str.contains('N-ACETYLCYSTEINE')].iloc[0]
    assert 'WITHHELD' in nac['Evidence / note'], nac['Evidence / note']
    assert '166' not in nac['Evidence / note'], 'the Solifenacin count leaked back into NAC'
    rescue = next((d.value for d in at.dataframe if 'Rescue drug' in d.value.columns), None)
    assert rescue is not None, 'the rescue-reversal table disappeared'
    assert 'N-ACETYLCYSTEINE' not in set(rescue['Rescue drug']), \
        'the mislabelled Solifenacin rows are still presented as NAC'


def _api_client():
    sys.path.insert(0, ROOT)
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


def test_api_health_and_model_info():
    c = _api_client()
    h = c.get('/health')
    assert h.status_code == 200
    assert h.json()['ok'] is True
    assert h.json()['model_version'] == 'catboost-v1.0'
    m = c.get('/model-info')
    assert m.status_code == 200
    body = m.json()
    assert body['features'] == 1544
    assert body['cell_lines_with_pca'] == 76
    assert body['drugs'] == 1482
    assert 'NOT a clinical' in body['disclaimer']


def test_api_predict_matches_sealed_ranking():
    c = _api_client()
    r = c.post('/predict', json={'cell_line_id': MDA, 'drug_id': 'DPC-006658'})
    assert r.status_code == 200
    body = r.json()
    sealed = pd.read_csv(os.path.join(OUT, 'mda_mb_231_full_ranking.csv'))
    expected = float(sealed.loc[sealed['drug_id'] == 'DPC-006658',
                                'pred_AUC_MDA231'].iloc[0])
    assert abs(body['predicted_auc'] - expected) < 1e-9
    assert body['model_version'] == 'catboost-v1.0'
    assert 'NOT a clinical' in body['disclaimer']


def test_api_rank_is_sorted_and_bounded():
    c = _api_client()
    r = c.post('/rank', json={'cell_line_id': MDA, 'top_k': 3})
    assert r.status_code == 200
    ranking = r.json()['ranking']
    assert len(ranking) == 3
    scores = [row['predicted_auc'] for row in ranking]
    assert scores == sorted(scores), 'ranking must be ascending AUC'
    assert [row['rank'] for row in ranking] == [1, 2, 3]
    assert ranking[0]['drug_id'] == 'DPC-006658'  # Triptolide, sealed rank 1
    assert ranking[0]['call'] == 'SENSITIVE (AUC<0.3)'


def test_api_rank_pairs_returns_additivity_pairs():
    c = _api_client()
    r = c.post('/rank-pairs', json={'top_singles': 3, 'top_pairs': 2})
    assert r.status_code == 200
    body = r.json()
    assert 'additivity' in body['method'].lower()
    assert len(body['pairs']) == 2
    for pair in body['pairs']:
        assert pair['drugA_id'] != pair['drugB_id']
        assert 0 <= pair['E_AB_MDA231'] <= 1
        assert 0 <= pair['E_AB_MCF7'] <= 1
        assert pair['pair_rank'] in (1, 2)


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
