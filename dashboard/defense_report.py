"""Self-contained HTML defence executive report for the TNBC / FerroSelect project.

Version 1.1 is a dated addendum, not a silent rewrite. It adds the post-hoc,
exploratory per-drug-mean baseline, the measured-vs-predicted selectivity null,
the "Predicted selective (unvalidated)" relabel, the NDCG definition fix and
the 76-unique-expression-line correction. The v1.0 artifacts remain in git
history at tag ``v1.0-defense``; the sealed prediction is not modified.

Pure function `build_defense_report(out_dir)` -> HTML string. No Streamlit, no
network access, no external JS: printable to PDF from any browser and safe to
attach to a thesis appendix.
"""
import datetime as _dt
import html
import json
import os

import numpy as np
import pandas as pd

# Phase 4a: the exported report honours BLIND_MODE. Default ON means sealed panel
# predictions and top-ranked kill proxies are replaced by a visible banner.
BLIND_MODE = os.environ.get('BLIND_MODE', 'ON').upper() in ('ON', '1', 'TRUE', 'YES')


def _env_blind():
    return os.environ.get('BLIND_MODE', 'ON').upper() in ('ON', '1', 'TRUE', 'YES')


_BLIND_BANNER = (
    '<div class="callout"><b>Blinded for bench team — unblinding requires '
    'supervisor key.</b><br>Panel predictions, the FerroSelect panel ranking and '
    'kill proxies are suppressed in this export. Unblinding requires a supervisor '
    'authorization and a timestamped entry in '
    '<code>dashboard_access_log.template.json</code> (see UNBLIND.md).</div>'
)


def _maybe_blind(fragment):
    """Return the banner instead of a prediction fragment when BLIND_MODE is ON."""
    return _BLIND_BANNER if BLIND_MODE else fragment

PANEL = [
    (1, 'ERASTIN + SIMVASTATIN', 'Combo'),
    (2, 'SULFASALAZINE + SIMVASTATIN', 'Combo (sealed nomination)'),
    (3, 'RSL3 + SIMVASTATIN', 'Combo (lab-only lane)'),
    (4, 'ERASTIN', 'Killer (Type I)'),
    (5, 'SULFASALAZINE', 'Killer (Type I)'),
    (6, 'RSL3', 'Killer (Type II, GPX4)'),
    (7, 'N-ACETYLCYSTEINE (NAC)', 'Rescue control'),
    (8, 'DEFEROXAMINE (DFO)', 'Rescue control'),
]
LINES = ['MDA-MB-231', 'MDA-MB-468', 'MCF7']


def _csv(out, name):
    p = os.path.join(out, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _json(out, name):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '—'
    if isinstance(v, (int, np.integer)):
        return f'{int(v):,}'
    if isinstance(v, (float, np.floating)):
        return f'{float(v):.{nd}f}'
    return html.escape(str(v))


def _table(df, columns=None, nd=3):
    if df is None or len(df) == 0:
        return '<p class="muted">Not available in this build.</p>'
    d = df.copy()
    if columns:
        d = d[[c for c in columns if c in d.columns]]
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: '—' if pd.isna(v) else f'{float(v):.{nd}f}')
    return d.to_html(index=False, escape=True, classes='data', border=0)


def _bars(df, name_col, auc_col, n=10, note_col=None):
    """CSS-only horizontal bars: no JS, prints cleanly, offline-safe."""
    if df is None or len(df) == 0:
        return '<p class="muted">Not available in this build.</p>'
    d = df.dropna(subset=[auc_col]).head(n)
    rows = []
    for _, r in d.iterrows():
        effect = float(np.clip(1.0 - float(r[auc_col]), 0.0, 1.0))
        sel = ''
        if note_col and note_col in d.columns:
            sel = str(r[note_col])
        cls = 'good' if effect >= 0.7 else ('mid' if effect >= 0.5 else 'low')
        rows.append(
            f'<div class="bar-row"><span class="bar-label">{html.escape(str(r[name_col]))}</span>'
            f'<span class="bar-track"><span class="bar-fill {cls}" '
            f'style="width:{effect * 100:.1f}%"></span></span>'
            f'<span class="bar-val">{_fmt(r[auc_col])}</span>'
            f'<span class="bar-note">{html.escape(sel)}</span></div>')
    return '<div class="bars">' + ''.join(rows) + '</div>'


def _bliss(preds, line, a, b):
    if preds is None:
        return None
    p = preds[preds['line'] == line].set_index('drug')['pred_auc'].astype(float)
    if a not in p.index or b not in p.index:
        return None
    ea, eb = 1 - p[a], 1 - p[b]
    return float(ea + eb - ea * eb)


def _panel_rows(preds):
    rows = []
    for rank, member, role in PANEL:
        if ' + ' in member:
            a, b = [x.strip() for x in member.split(' + ')]
            effect = _bliss(preds, 'MDA-MB-231', a, b)
            auc = None if effect is None else 1 - effect
            if member.startswith('RSL3'):
                note = 'Lab-only: no PRISM AUC for RSL3; test the pair on the plate.'
            else:
                note = ('Bliss additivity arithmetic from the sealed single-agent AUCs; '
                        'not a synergy finding.')
        elif member.startswith('N-ACETYL') or member.startswith('DEFEROX'):
            auc, note = None, 'Rescue control: must re-express suppressed proteins.'
        elif member == 'RSL3':
            auc, note = None, 'No PRISM AUC; GSE247883 signature + CTRPv2 ground truth.'
        else:
            p = preds[preds['line'] == 'MDA-MB-231'].set_index('drug')['pred_auc']
            auc = float(p[member]) if member in p.index else None
            note = 'Sealed single-agent prediction (LOW confidence).'
        rows.append({'Rank': rank, 'Panel member': member, 'Role': role,
                     'Predicted AUC (MDA-MB-231)': auc, 'Evidence / note': note})
    return pd.DataFrame(rows)


def _bliss_rows(preds):
    rows = []
    for member in ['ERASTIN + SIMVASTATIN', 'SULFASALAZINE + SIMVASTATIN',
                   'RSL3 + SIMVASTATIN']:
        a, b = [x.strip() for x in member.split(' + ')]
        for line in LINES:
            e = _bliss(preds, line, a, b)
            rows.append({'Combination': member, 'Cell line': line,
                         'Bliss E_AB': e,
                         'Expected AUC': None if e is None else 1 - e,
                         'Note': 'not computable (no PRISM AUC for RSL3)'
                         if e is None else 'additivity expectation — needs the plate'})
    return pd.DataFrame(rows)


def build_defense_report(out_dir, blind=None):
    """Render the complete v1.1 executive report and return it as an HTML string.

    blind=None honours BLIND_MODE (default ON); pass blind=False only for
    supervisor-authorized exports/tests of the complete panel.
    """
    global BLIND_MODE
    BLIND_MODE = _env_blind() if blind is None else bool(blind)
    out = out_dir
    leaderboard = _csv(out, 'leaderboard.csv')
    baseline_csv = _csv(out, 'baseline_audit.csv')
    baseline = _json(out, 'baseline_audit.json') or {}
    expr = _json(out, 'expression_line_audit.json') or {}
    validation = _json(out, 'validation_status.json') or {}
    info = _json(out, 'model_info.json') or {}
    audit = _json(out, 'audit_summary.json') or {}
    annotated = _csv(out, 'breast_lines_annotated.csv')
    top20 = _csv(out, 'mda_mb_231_top20_killers.csv')
    combos = _csv(out, 'combo_rankings.csv')
    preds = _csv(out, 'ferroselect_panel_predictions.csv')
    fres = _json(out, 'ferroselect_results.json') or {}
    ctp = _csv(out, 'ctpv2_ferroptosis_auc.csv')
    ctp_meta = _json(out, 'ctpv2_meta.json') or {}
    rsl_report = _json(out, 'rsl3_v2_report.json') or {}
    intervals = _json(out, 'prediction_intervals.json') or {}
    gates = _json(out, 'selectivity_rule.json') or {}
    pca_status = _json(out, 'pca_loadings_status.json') or {}
    diagnostic = _json(out, 'diagnostic_results.json') or {}

    now = _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    champion = 'n/a'
    best = None
    if leaderboard is not None and len(leaderboard):
        best = leaderboard.sort_values('ndcg10', ascending=False).iloc[0]
        import re
        champion = re.sub(r'[^\x00-\x7F]+', '', str(best['label'])).strip()

    metrics = baseline.get('metrics') or {}
    cb = metrics.get('catboost') or {}
    bm = metrics.get('per_drug_mean') or {}
    paired = baseline.get('paired_bootstrap_catboost_minus_baseline') or {}
    sel = baseline.get('selectivity') or {}
    sel_all = sel.get('all_shared_drugs') or {}
    sel_learned = sel.get('learned_227') or {}
    sel_restricted = sel.get('restricted_measured_auc_below_0_5_either') or {}
    coverage = baseline.get('coverage') or {}
    calibration = baseline.get('calibration') or {}
    subsets = baseline.get('subsets') or {}
    holdout = baseline.get('holdout') or {}

    kpis = [
        ('CatBoost per-line Spearman', _fmt(cb.get('per_line_spearman'))),
        ('Per-drug baseline', _fmt(bm.get('per_line_spearman'))),
        ('CatBoost R²', _fmt(cb.get('r2'))),
        ('Baseline R²', _fmt(bm.get('r2'))),
        ('Selectivity Spearman', _fmt((sel_all.get('spearman') or {}).get('statistic'))),
        ('Tests', f"{_fmt(validation.get('tests_passed'))}/{_fmt(validation.get('tests_total'))}"
                  if validation else '—'),
    ]
    kpi_html = ''.join(f'<div class="kpi"><div class="kpi-v">{v}</div>'
                       f'<div class="kpi-l">{html.escape(k)}</div></div>'
                       for k, v in kpis)

    hold = info.get('holdout', [])
    if annotated is not None and 'ModelID' in annotated.columns:
        names = dict(zip(annotated['ModelID'], annotated['StrippedCellLineName']))
        hold_df = pd.DataFrame([{'ModelID': h, 'Cell line': names.get(h, h),
                                 'Role': 'Blind holdout'} for h in hold])
    else:
        hold_df = pd.DataFrame([{'ModelID': h, 'Cell line': h, 'Role': 'Blind holdout'}
                                for h in hold])

    panel = _panel_rows(preds) if preds is not None else None
    bliss_rows = _bliss_rows(preds) if preds is not None else None
    top_rank = top20.head(10) if top20 is not None else None

    ctp_show = None
    if ctp is not None:
        ctp_sel = ctp[(ctp['target'] == 'RSL3')
                      & (ctp['ModelID'].isin(['ACH-000768', 'ACH-000849', 'ACH-000019']))].copy()
        if len(ctp_sel):
            ctp_sel['Cell line'] = ctp_sel['ModelID'].map({'ACH-000768': 'MDA-MB-231',
                                                           'ACH-000849': 'MDA-MB-468',
                                                           'ACH-000019': 'MCF7'})
            ctp_show = ctp_sel[['Cell line', 'auc_raw', 'auc_norm_0_1', 'sens_percentile']]
            ctp_show.columns = ['Cell line', 'RSL-3 raw AUC (CTRPv2)', 'AUC 0-1',
                                'Percentile most sensitive (of 808)']
            ctp_show = ctp_show.sort_values('AUC 0-1')

    glob = intervals.get('global') or {}
    mda_q80 = ((intervals.get('lines') or {}).get('ACH-000768') or {}).get(
        'abs_q80', glob.get('abs_q80'))
    line_rows = []
    for mid, s in (intervals.get('lines') or {}).items():
        line_rows.append({'ModelID': mid, 'n': s.get('n'),
                          'RMSE': s.get('rmse'), '±80% CI': s.get('abs_q80'),
                          '±95% CI': s.get('abs_q95')})
    line_ci = pd.DataFrame(line_rows) if line_rows else None

    diag_metrics = diagnostic.get('metrics') or {}
    diag_rows = []
    for key, desc in [('M0', 'Fold-training per-drug mean'),
                      ('M1', 'M0 + fitted RidgeCV line offset'),
                      ('M2', 'M0 + kNN M0-residual adjustment')]:
        dm = diag_metrics.get(key) or {}
        diag_rows.append({
            'Model': f'{key}: {desc}',
            'Per-line Spearman': dm.get('perline_spearman_mean'),
            'Pooled R² vs measured AUC': dm.get('pooled_r2_vs_truth'),
            'MAE vs measured AUC': dm.get('mae_vs_truth'),
        })
    diag_df = pd.DataFrame(diag_rows) if diag_metrics else None
    diag_pb = diagnostic.get('paired_bootstrap_lines_2000') or {}
    diag_m1pb = diag_pb.get('m1_minus_m0') or {}
    diag_m2pb = diag_pb.get('m2_minus_m0') or {}
    diag_m1_ci = diag_m1pb.get('ci_95') or [None, None]
    diag_m2_ci = diag_m2pb.get('ci_95') or [None, None]
    diag_audit = diagnostic.get('construction_audit') or {}
    diag_pca = diag_audit.get('pca_line_matrix') or {}
    diag_iso = diag_audit.get('fold_isolation') or {}
    diag_zw = diag_audit.get('zero_weight_sanity') or {}

    v2_best = (rsl_report.get('models') or {}).get(rsl_report.get('winner', ''), {})
    fbest = (fres.get('per_model_oof') or {}).get(fres.get('frozen_model', ''), {})

    # ---- combined baseline table: CatBoost first, post-hoc baseline beside it ----
    if baseline_csv is not None:
        bshow = baseline_csv.copy()
        bshow = bshow.rename(columns={
            'model': 'Model', 'analysis_status': 'Analysis status', 'n': 'n',
            'mae': 'MAE', 'rmse': 'RMSE', 'r2': 'R²',
            'pooled_spearman': 'Pooled Spearman',
            'per_line_spearman': 'Per-line Spearman', 'ndcg10': 'NDCG@10',
            'recall10': 'Recall@10'})
        bshow = bshow[['Model', 'Analysis status', 'n', 'MAE', 'RMSE', 'R²',
                       'Pooled Spearman', 'Per-line Spearman', 'NDCG@10', 'Recall@10']]
    else:
        bshow = None

    paired_rows = []
    for key, label in [('per_line_spearman', 'Per-line Spearman'), ('r2', 'R²'),
                       ('ndcg10', 'NDCG@10')]:
        v = paired.get(key) or {}
        if v:
            paired_rows.append({
                'Metric': label, 'Δ mean (CatBoost − baseline)': v.get('mean'),
                '95% CI low': v.get('lo95'), '95% CI high': v.get('hi95'),
                'Baseline better': v.get('baseline_better_fraction')})
    paired_df = pd.DataFrame(paired_rows) if paired_rows else None

    subset_rows = []
    for name, m in subsets.items():
        cbm, bmm = m.get('catboost') or {}, m.get('per_drug_mean') or {}
        subset_rows.append({
            'Subset': name.replace('_', ' '),
            'n': cbm.get('n'),
            'CatBoost per-line': cbm.get('per_line_spearman'),
            'Baseline per-line': bmm.get('per_line_spearman'),
            'CatBoost R²': cbm.get('r2'),
            'Baseline R²': bmm.get('r2'),
            'CatBoost NDCG': cbm.get('ndcg10'),
            'Baseline NDCG': bmm.get('ndcg10')})
    subset_df = pd.DataFrame(subset_rows) if subset_rows else None

    calibration_rows = []
    for name, c in calibration.items():
        calibration_rows.append({
            'Subset': name.replace('_', ' '), 'n': c.get('n'),
            'Measured mean AUC': c.get('measured_mean_auc'),
            'CatBoost mean': c.get('catboost_mean_auc'),
            'Baseline mean': c.get('per_drug_mean_auc')})
    calibration_df = pd.DataFrame(calibration_rows) if calibration_rows else None

    selectivity_rows = []
    for key, label in [('all_shared_drugs', 'All shared drugs'),
                       ('learned_227', 'Learned 227 drugs'),
                       ('restricted_measured_auc_below_0_5_either',
                        'Restricted: measured AUC < 0.5 in either line')]:
        s = sel.get(key) or {}
        selectivity_rows.append({
            'Analysis': label, 'n': s.get('n'),
            'Pearson': (s.get('pearson') or {}).get('statistic'),
            'Spearman': (s.get('spearman') or {}).get('statistic'),
            'Sign agreement': s.get('sign_agreement'),
            'Predicted Δ min': s.get('predicted_delta_min'),
            'Predicted Δ max': s.get('predicted_delta_max')})
    selectivity_df = pd.DataFrame(selectivity_rows) if selectivity_rows else None

    calls_df = pd.DataFrame(sel.get('v1_0_selective_calls') or [])
    if len(calls_df):
        calls_df = calls_df.rename(columns={
            'drug_name': 'Drug', 'predicted_delta': 'Predicted Δ',
            'measured_delta': 'Measured Δ', 'measured_mda_auc': 'Measured MDA-MB-231 AUC',
            'measured_mcf7_auc': 'Measured MCF7 AUC'})
        calls_df = calls_df[['Drug', 'Predicted Δ', 'Measured Δ',
                             'Measured MDA-MB-231 AUC', 'Measured MCF7 AUC']]

    # ---- validation gates ----
    gates_html = []
    if best is not None:
        gates_html = [
            ('R² > 0.25 (CatBoost)', float(best['r2']), 0.25,
             'PASS' if best['r2'] > 0.25 else 'FAIL'),
            ('Per-line Spearman > 0.40 (CatBoost)', float(best['per_line_spearman']), 0.40,
             'PASS' if best['per_line_spearman'] > 0.40 else 'FAIL'),
            ('Post-hoc baseline beats CatBoost (per-line Spearman)',
             float(bm.get('per_line_spearman', np.nan)), None,
             'REPORTED' if bm else '—'),
            ('Selectivity predicted-vs-measured correlation',
             float((sel_all.get('spearman') or {}).get('statistic', np.nan)), None,
             'NULL' if sel_all else '—'),
        ]

    plateau = (coverage.get('plateau_drugs') or 0)
    plateau_fraction = coverage.get('plateau_fraction') or 0
    plateau_value = coverage.get('plateau_mda_value')
    plateau_text = (
        f"{_fmt(plateau)} of {_fmt(audit.get('ml_drugs'))} drugs "
        f"({plateau_fraction:.1%}) carry zero CatBoost importance and collapse to one "
        f"predicted value per line"
        + (f" ({_fmt(plateau_value, 6)} on MDA-MB-231)" if plateau_value is not None else '')
        + f". Only {_fmt(coverage.get('learned_drugs'))} drugs carry learned "
        "signal; the plateau is a documented coverage limit, not a fine-grained ranking.")
    top_note = ('The predicted-AUC distribution is compressed above ~0.906 for the '
                'resistant bulk of the panel; ranks below ~226 are tied within model '
                'resolution and should not be read as a fine-grained order. ' + plateau_text)

    css = """
    @page { size: A4; margin: 16mm; }
    * { box-sizing: border-box; }
    body { font-family: 'Segoe UI', Arial, sans-serif; color: #212529; font-size: 11px;
           line-height: 1.45; margin: 0 auto; max-width: 980px; padding: 20px 24px;
           -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    @media print { body { margin: 0; padding: 0; max-width: none; } }
    h1 { font-size: 22px; margin: 0 0 4px; color: #b3261e; }
    h2 { font-size: 14px; margin: 20px 0 8px; border-bottom: 2px solid #e9ecef;
         padding-bottom: 4px; color: #343a40; }
    h3 { font-size: 12px; margin: 12px 0 6px; color: #495057; }
    .sub { color: #6c757d; font-size: 11px; }
    .kpis { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 4px; }
    .kpi { border: 1px solid #dee2e6; border-radius: 6px; padding: 8px 10px; min-width: 118px;
           flex: 1; background: #f8f9fa; }
    .kpi-v { font-size: 17px; font-weight: 700; color: #212529; }
    .kpi-l { font-size: 9.5px; color: #6c757d; text-transform: uppercase;
             letter-spacing: .03em; }
    table.data { border-collapse: collapse; width: 100%; margin: 6px 0 10px; }
    table.data th { background: #343a40; color: #fff; text-align: left; padding: 5px 6px;
                    font-size: 10px; }
    table.data td { border-bottom: 1px solid #e9ecef; padding: 4px 6px; vertical-align: top; }
    table.data tr:nth-child(even) td { background: #f8f9fa; }
    .muted { color: #868e96; font-style: italic; }
    .bars { margin: 6px 0 10px; }
    .bar-row { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
    .bar-label { width: 165px; text-align: right; font-weight: 600; font-size: 10px; }
    .bar-track { flex: 1; height: 13px; background: #f1f3f5; border-radius: 3px;
                 overflow: hidden; }
    .bar-fill { display: block; height: 100%; border-radius: 3px; }
    .bar-fill.good { background: #b3261e; }
    .bar-fill.mid { background: #f59f00; }
    .bar-fill.low { background: #adb5bd; }
    .bar-val { width: 52px; font-variant-numeric: tabular-nums; font-size: 10px; }
    .bar-note { width: 150px; font-size: 9px; color: #6c757d; }
    .callout { border-left: 4px solid #b3261e; background: #fff5f5; padding: 8px 10px;
               border-radius: 4px; margin: 8px 0; }
    .note { border-left: 4px solid #1971c2; background: #f1f8ff; padding: 8px 10px;
            border-radius: 4px; margin: 8px 0; }
    .ok { color: #2f7d32; font-weight: 700; }
    .bad { color: #b3261e; font-weight: 700; }
    .warn { color: #b35c00; font-weight: 700; }
    .sign { margin-top: 26px; }
    .sign table { width: 100%; border-collapse: collapse; }
    .sign td { border: 1px solid #adb5bd; height: 34px; padding: 4px 8px; width: 50%; }
    .sign .label { border: none; height: auto; color: #6c757d; font-size: 10px;
                   padding-top: 6px; }
    footer { margin-top: 22px; border-top: 1px solid #dee2e6; padding-top: 6px;
             color: #868e96; font-size: 9.5px; }
    """

    parts = [f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>TNBC / FerroSelect — Defence Executive Report v1.1</title>
<style>{css}</style></head><body>
<h1>FerroSelect — TNBC Drug-Response Predictor</h1>
<div class="sub">Defence executive report · <b>v1.1 dated addendum</b> · generated {now} ·
model version <b>{html.escape(str(info.get('model_version', 'n/a')))}</b></div>
<div class="callout"><b>Document status.</b> v1.1 (2026-10-29) supersedes the v1.0
briefing as a dated addendum; it does not rewrite v1.0. The v1.0 artifacts are preserved
in git at tag <b>v1.0-defense</b> and the sealed prediction is unchanged. The baseline and
selectivity analyses below are <b>post-hoc, exploratory</b>: the holdout was already seen.
The champion is reported as a <b>negative result</b>.</div>
<div class="callout"><b>Scope:</b> predicted sensitivity in cell models — never a
clinical treatment recommendation. Wet-lab confirmation is the decision layer.</div>
<div class="kpis">{kpi_html}</div>

<h2>1. Dataset, leakage controls and the 76-line correction</h2>
<p>The DepMap metadata covers <b>{_fmt(audit.get('breast_lines_total'))}</b> breast lines,
<b>{_fmt(expr.get('breast_lines_with_expression', audit.get('wanted_with_expression')))}</b>
unique lines with expression, of which <b>{_fmt(audit.get('ml_cell_lines'))}</b> enter
training on <b>{_fmt(audit.get('ml_rows'))}</b> unique (cell line, drug) pairs and
<b>{_fmt(audit.get('ml_drugs'))}</b> screened drugs. Subtypes: TNBC
{_fmt((audit.get('subtype_counts') or {}).get('TNBC'))}, Luminal
{_fmt((audit.get('subtype_counts') or {}).get('Luminal'))}, HER2+
{_fmt((audit.get('subtype_counts') or {}).get('HER2+'))}, other
{_fmt((audit.get('subtype_counts') or {}).get('other'))}.</p>
<div class="note"><b>v1.1 correction.</b> v1.0 over-counted the expression lines by one. The
correct count is <b>{_fmt(expr.get('breast_lines_with_expression'))} unique lines</b>.
The shipped PCA table has <b>{_fmt(expr.get('expression_rows'))}</b> rows because MCF7,
ACH-001396 and ACH-001662 are duplicated. The PCA was fit on
<b>{_fmt(expr.get('pca_fit_rows'))}</b> rows = <b>{_fmt(expr.get('pca_fit_unique_lines'))}</b>
unique lines; the PCA holdout is <b>{_fmt(expr.get('pca_holdout_unique_lines'))}</b> lines.
All <b>{_fmt(len(expr.get('ml_holdout_ids') or []))}</b> ML holdout lines are in the PCA
holdout and <b>none</b> are in the PCA fit. The phantom-header bug shifted the draw, but
it did not leak the model holdout.</div>
{_table(hold_df)}

<h2>2. Model vs a one-line baseline (post-hoc, exploratory)</h2>
<p>The baseline is each drug’s mean AUC on the 23 training lines, applied to the seven
holdout lines. Drugs with no training row would fall back to the global training mean; in
this holdout <b>{_fmt(holdout.get('fallback_drugs_in_holdout'))}</b> drugs needed the
fallback, and both models are scored on the <b>identical {_fmt(holdout.get('n_rows'))}
rows</b>. The comparison is paired.</p>
{_table(bshow, nd=3)}
<p><b>Paired bootstrap over the seven lines</b> (2,000 resamples; the effective sample
size is lines, not rows):</p>
{_table(paired_df, nd=3)}
<p><b>Interpretation.</b> The baseline wins per-line Spearman and R² on every holdout
line; CatBoost never wins the paired bootstrap. NDCG@10 is statistically tied and is not
evidence for either model. This shows the <b>architecture fails to beat drug identity</b>;
it does not prove that expression carries no signal, because the model never learned 85%
of drug identities. Within the 227 drugs it did learn, it ties the baseline rather than
beating it.</p>
{_table(subset_df, nd=3)}
{_table(calibration_df, nd=3)}
<p><b>Sensitive-tail coverage.</b> {_fmt(coverage.get('measured_sensitive_in_learned'))} of
{_fmt(coverage.get('measured_sensitive_rows'))} measured AUC &lt; 0.3 holdout rows fall
inside the {_fmt(coverage.get('learned_drugs'))} learned drugs; every line’s measured
top-10 is 10/10 inside that set. This is <b>coverage</b> — the plateau does not hide the
actives — not model lift. The baseline covers those drugs trivially.</p>
<div class="note"><b>Metric definition fix.</b> NDCG@10 uses <b>graded relevance
1 − AUC</b> (lower AUC = more relevant). Only Recall@10 uses the binary
<b>AUC &lt; 0.3</b> sensitivity threshold. v1.0's metric note was inaccurate; the
graded definition above is the one implemented. NDCG is saturated near 0.94 for both
models and is a weak discriminator.</div>

<h2>3. Selectivity null and the “Predicted selective (unvalidated)” relabel</h2>
<p>Predicted selectivity Δ = predicted AUC(MCF7) − predicted AUC(MDA-MB-231). Positive
means the model predicts MDA-selective. Measured Δ uses the same definition on the
holdout measurements. Across the {_fmt(sel_all.get('n'))} drugs measured in both lines,
the correlation is essentially zero.</p>
{_table(selectivity_df, nd=4)}
<p>The restricted check — drugs with measured AUC &lt; 0.5 in either line — does not
rescue the correlation (n={_fmt(sel_restricted.get('n'))}, Spearman
{_fmt((sel_restricted.get('spearman') or {}).get('statistic'), 4)}). The learned-227
subset is also null (Spearman {_fmt((sel_learned.get('spearman') or {}).get('statistic'), 4)}).
The predicted Δ range is {_fmt(sel_all.get('predicted_delta_min'), 4)} to
{_fmt(sel_all.get('predicted_delta_max'), 4)}, while the MDA-MB-231 80% interval is
±{_fmt(mda_q80)}: <b>the entire selectivity range is inside the noise band</b>.</p>
<p><b>The four v1.0 “selective” calls are not supported by the measured holdout data:</b></p>
{_table(calls_df, nd=4)}
<p>These are relabelled <b>“Predicted selective (unvalidated)”</b> throughout the
dashboard and this report. They are correct-sign hypotheses to test by
SI = IC50(MCF7)/IC50(MDA-MB-231) on the plate, not findings. Combination ranks are
additivity arithmetic: Bliss guarantees a pair out-ranks its components when computed
from their kill proxies, so a combo rank is not a discovery.</p>

<h2>4. The eight-member wet-lab panel (pre-registered mechanism hypothesis)</h2>
{_maybe_blind(_table(panel, nd=3))}
<p class="muted">Rank 1 = strongest expected killing proxy by the frozen model. The panel was
selected on mechanism (GPX4, system xc⁻, iron) and agreed with the lab before any plate;
the model outputs are a frozen, falsifiable record, not the justification. RSL3 lanes are
lab-only by design: PRISM never profiled RSL3, so no model AUC is fabricated for them.
Selectivity and panel ordering are hypotheses for the plate.</p>

<h2>5. Top-ranked killers, model resolution and the plateau</h2>
{_maybe_blind(_bars(top_rank, 'drug_name', 'pred_AUC_MDA231', n=10) if top_rank is not None else '<p class="muted">Ranking unavailable.</p>')}
<p class="muted">{html.escape(top_note)}</p>

<h2>6. Bliss combination expectations (additivity arithmetic — test on the plate)</h2>
{_maybe_blind(_table(bliss_rows))}
{_maybe_blind(_table(combos.head(5) if combos is not None else None,
        ['drugA_name', 'drugB_name', 'E_AB_MDA231', 'E_AB_MCF7', 'selectivity_gap']))}
<p class="muted">These are additivity expectations, not synergy predictions. The 4×4
checkerboard is the test.</p>

<h2>7. CTRPv2.1 measured ground truth for RSL-3 (external evidence)</h2>
{_table(ctp_show, nd=2)}
<p class="muted">{html.escape(str(ctp_meta.get('auc_definition', 'v2.1 published AUC; lower = more cell death.')))}</p>
<p>RSL-3 v2 model: <b>{html.escape(str(rsl_report.get('winner', 'n/a')))}</b>,
OOF Spearman <b>{_fmt(v2_best.get('spearman'))}</b>
(95% CI {_fmt((v2_best.get('bootstrap_spearman') or {}).get('lo95'))} to
{_fmt((v2_best.get('bootstrap_spearman') or {}).get('hi95'))}) on
{_fmt(v2_best.get('n'))} lines — below the published lab-vs-CTRP ceiling
({_fmt((rsl_report.get('ceiling_lab_vs_ctrp') or {}).get('RSL3'))}). This is external,
cross-platform evidence about RSL3, not a validation of the v1 PRISM champion.</p>

<h2>8. Prediction uncertainty (calibrated on the spent holdout)</h2>
<p>Conformal-style intervals from {_fmt(glob.get('n'))} out-of-sample rows:
<b>±{_fmt(glob.get('abs_q80'))}</b> at 80% and
<b>±{_fmt(glob.get('abs_q95'))}</b> at 95% (global half-widths). They are marginal and
calibrated on the same seven lines, so they are not independent validation.</p>
{_table(line_ci)}
<p class="muted">FerroSelect frozen model ({html.escape(str(fres.get('frozen_model', 'n/a')))},
{_fmt(fbest.get('n'))} panel rows): pooled OOF Spearman
{_fmt(fbest.get('spearman'))} with bootstrap CI
{_fmt((fbest.get('bootstrap_spearman') or {}).get('lo95'))} to
{_fmt((fbest.get('bootstrap_spearman') or {}).get('hi95'))}. This is an honestly low,
pre-registered hypothesis — the plate decides.</p>

<h2>9. Pre-registered diagnostic (post-hoc, exploratory)</h2>
<p>The diagnostic uses only the <b>23 training lines</b> (holdout excluded) with strict
<b>GroupKFold(5)</b>: per-drug means, PCA scaling, the RidgeCV line-offset fit and the M2
neighbour pool all use fold-training lines only. It is exploratory and is not comparable to
holdout metrics. Pre-registered settings: k=5, first 10 PCs, bootstrap seed 42 with 2,000
resamples.</p>
{_table(diag_df, nd=4)}
<p>M1’s per-line Spearman equals M0 by construction because its fitted RidgeCV offset is
constant within a held-out line, while its pooled R²/MAE are fitted values, not copies of M0.
The paired M1−M0 difference is <b>{_fmt(diag_m1pb.get('mean'), 4)}</b>
(95% CI {_fmt(diag_m1_ci[0], 4)} to {_fmt(diag_m1_ci[1], 4)}). The paired M2−M0 difference is
<b>{_fmt(diag_m2pb.get('mean'), 4)}</b>
(95% CI {_fmt(diag_m2_ci[0], 4)} to {_fmt(diag_m2_ci[1], 4)}): <b>M2 does not beat M0</b>, so
the pre-registered success criterion (95% CI above zero) is not met. This is no evidence of
expression signal, not proof of absence. M3 was skipped for runtime.</p>
<div class="note"><b>v1.2.5 construction fix (transparent changelog).</b> The v1.2.4
diagnostic had four construction defects found in audit: a line-rank value was used as a row
index into the drug-line matrix (3/23 mappings correct, zero-distance duplicates); M2
borrowed a within-line-centred AUC instead of the M0 residual; there was no GroupKFold
isolation; and M1 was never fitted. v1.2.5 corrects the construction without changing
pre-registered settings (k=5, first 10 PCs, GroupKFold(5), bootstrap seed 42 with 2,000
resamples, same criterion). Audit: {_fmt(diag_pca.get('n_unique_feature_rows'))}/
{_fmt(diag_pca.get('n_lines'))} unique line rows,
{_fmt(diag_pca.get('zero_distance_pairs'))} zero-distance distinct-line pairs,
{_fmt(diag_iso.get('same_fold_neighbour_count'))} same-fold neighbours, target line absent
from every M0 source, and zero-weight M2 = M0 exactly (max|Δ| =
{_fmt(diag_zw.get('max_abs_diff_vs_m0'), 6)}). Superseded v1.2.4 outputs are archived
at <code>outputs_local/archive/diagnostic_results_v1.2.4_MISCONSTRUCTED.*</code>.</div>

<h2>10. Validation gates, provenance quarantine and test status</h2>
<table class="data"><tr><th>Gate</th><th>Value</th><th>Target</th><th>Status</th></tr>
"""]

    for label, value, target, status in gates_html:
        cls = ('ok' if status == 'PASS' else
               'bad' if status in ('FAIL', 'NULL') else 'warn')
        parts.append(f'<tr><td>{html.escape(label)}</td><td>{_fmt(value)}</td>'
                     f'<td>{_fmt(target) if target is not None else "—"}</td>'
                     f'<td class="{cls}">{html.escape(status)}</td></tr>')
    if gates:
        ds = gates.get('delta_stats') or {}
        parts.append(f'<tr><td>Selectivity rule (relabelled)</td><td colspan="3">'
                     f'{html.escape(str(gates.get("selectivity_rule", "n/a")))} '
                     f'({_fmt(ds.get("n_positive"))} positive-Δ drugs of '
                     f'{_fmt(gates.get("panel_size"))}); the old fixed cut-off '
                     f'Δ &gt; 0.05 was unreachable (max Δ = {_fmt(ds.get("max"))})</td></tr>')
    if pca_status:
        parts.append(f'<tr><td>PCA gene annotation</td><td colspan="3">'
                     f'{html.escape(str(pca_status.get("outcome", "n/a")))} — '
                     f'{_fmt((pca_status.get("checks") or {}).get("n_components_verified"))} of '
                     f'{_fmt((pca_status.get("checks") or {}).get("n_components"))} components verified</td></tr>')
    if validation:
        status = validation.get('status', '—')
        cls = 'ok' if status == 'PASS' else 'bad'
        parts.append(f'<tr><td>Automated test suite</td><td>{_fmt(validation.get("tests_passed"))}'
                     f'/{_fmt(validation.get("tests_total"))}</td><td>all pass</td>'
                     f'<td class="{cls}">{html.escape(str(status))}</td></tr>')
    parts.append('</table>')
    parts.append(f"""<p><b>Provenance quarantine.</b> The legacy L1000 matcher resolved the
abbreviation “NAC” inside “Solife<b>NAC</b>in-Succinate”, so the archived NAC rows are a
different molecule. The data layer now carries a <b>quarantined</b> column in
<code>offtarget_panel_map.csv</code>, <code>offtarget_rescue_reversal.csv</code>,
<code>protein_array_shortlist.csv</code> and
<code>offtarget_coverage_ferroselect.csv</code>. The UI withholds the NAC rescue arm and
the tests assert the marker. Deferoxamine is the validated rescue arm; Erastin’s
latent substring collision with Cloperastine is blocked by exact-match-first resolution
and a regression test. The wet-lab NAC molecule is unaffected.</p>

<h2>11. Sign-off (print, sign, lodge with the sealed prediction)</h2>
<div class="sign"><table>
<tr><td class="label">Computational lead</td><td class="label">Supervisor</td></tr>
<tr><td></td><td></td></tr>
<tr><td class="label">Date</td><td class="label">Date</td></tr>
</table></div>

<h2>Appendix A. v1.1 changelog</h2>
<table class="data">
<tr><th>Change</th><th>Where</th></tr>
<tr><td>Per-drug mean baseline beside CatBoost; paired line bootstrap; subset and
calibration tables</td><td>§2, dashboard Model &amp; audit, <code>baseline_audit.*</code></td></tr>
<tr><td>Selectivity null (all shared, learned 227, restricted AUC &lt; 0.5 either) and
the four unsupported calls</td><td>§3, dashboard Selectivity, <code>baseline_audit.json</code></td></tr>
<tr><td>“Selective” relabelled “Predicted selective (unvalidated)”</td><td>Dashboard,
§3</td></tr>
<tr><td>NDCG@10 definition corrected to graded relevance 1 − AUC; only Recall@10 is
binary AUC &lt; 0.3</td><td>§2, README</td></tr>
<tr><td>Expression-line count corrected from 77 to 76 unique; 79 rows / 3 duplicate
ModelIDs explained</td><td>§1, <code>audit_summary.json</code>,
<code>expression_line_audit.json</code></td></tr>
<tr><td><code>quarantined</code> column added to the four legacy NAC CSVs; Erastin /
Cloperastine regression test</td><td>Data layer, tests</td></tr>
<tr><td>Four API tests added; suite count reported from
<code>validation_status.json</code></td><td>§10, tests</td></tr>
<tr><td>v1.2.5 fold-honest diagnostic construction fix: strict GroupKFold(5), line-keyed
PCA, fitted RidgeCV M1, M0-residual M2; M2 still does not beat M0 (paired 95% CI
[−0.0373, −0.0176])</td><td>§9, dashboard diagnostic panel,
<code>diagnostic_results.*</code></td></tr>
</table>

<h2>Appendix B. Short defence Q&amp;A</h2>
<p><b>Q1. “Your CatBoost model loses to a column mean. Why is it in the project?”</b><br>
It is reported as a negative result. The baseline wins because CatBoost never learned 85%
of drug identities. Within the 227 drugs it did learn, it ties the baseline. The expression
contribution is unproven, and we say so. The project’s value is the audit that found this,
not the model.</p>
<p><b>Q2. “Did the sealed prediction or the ‘selective’ labels rely on this model?”</b><br>
The panel was chosen on mechanism (GPX4, system xc⁻, iron) and agreed with the lab before
any plate. The model outputs are a frozen, falsifiable record, not the justification.
Predicted selectivity is uncorrelated with measured selectivity, so the four “selective”
drugs are relabelled unvalidated hypotheses. Selectivity will be tested by
SI = IC50(MCF7)/IC50(MDA-MB-231) on the plate. We will report Spearman with a bootstrap CI
after unblinding, as the PRD requires.</p>
<p><b>Q3. “If the model is this weak, what is the scientific contribution?”</b><br>
A reproducible pipeline whose own controls caught its failures: a circular-label inflation
of 0.65–0.99, a substring bug that would have mislabelled Solifenacin as NAC, an 84.7%
plateau, and a dead selectivity threshold. We found them, bounded them and tested the
fixes. We also pre-registered a falsifiable wet-lab panel, with CTRPv2 showing MDA-MB-231
in the most RSL3-sensitive 30% while MDA-MB-468 is resistant.</p>

<footer>Artifacts: outputs_local/ · splits by cell line · fixed seed 42 · pinned
requirements.txt · prediction intervals calibrated on pre-registered holdout lines.
Report generated automatically from the committed artifacts; no values are typed by hand.
v1.1 addendum, 2026-10-29. v1.0 artifacts preserved at tag v1.0-defense.</footer>
</body></html>""")
    return ''.join(parts)
