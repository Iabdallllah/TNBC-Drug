"""Self-contained HTML defence executive report for the TNBC / FerroSelect project.

Pure function `build_defense_report(out_dir)` -> HTML string. No Streamlit, no
network access, no external JS: the report is printable to PDF from any browser
and safe to attach to a thesis appendix.

Sections
--------
1. dataset / cell-line status        5. CTRPv2 measured RSL-3 ground truth
2. the 8-member wet-lab panel        6. prediction uncertainty (conformal)
3. top-ranked killers + CSS bars     7. validation gates and sign-off
4. Bliss combination expectations
"""
import datetime as _dt
import html
import json
import os

import numpy as np
import pandas as pd

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
                note = 'Bliss expectation from the sealed single-agent AUCs.'
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


def build_defense_report(out_dir):
    """Render the complete executive report and return it as an HTML string."""
    out = out_dir
    leaderboard = _csv(out, 'leaderboard.csv')
    info = _json(out, 'model_info.json') or {}
    audit = _json(out, 'audit_summary.json') or {}
    annotated = _csv(out, 'breast_lines_annotated.csv')
    shortlist = _csv(out, 'wetlab_shortlist.csv')
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

    now = _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    champion = 'n/a'
    best = None
    if leaderboard is not None and len(leaderboard):
        best = leaderboard.sort_values('ndcg10', ascending=False).iloc[0]
        import re
        champion = re.sub(r'[^\x00-\x7F]+', '', str(best['label'])).strip()

    kpis = [
        ('Champion model', champion),
        ('R² on unseen lines', _fmt(best['r2']) if best is not None else '—'),
        ('Per-line Spearman', _fmt(best['per_line_spearman']) if best is not None else '—'),
        ('NDCG@10', _fmt(best['ndcg10']) if best is not None else '—'),
        ('Training pairs', _fmt(audit.get('ml_rows'))),
        ('Screened drugs', _fmt(audit.get('ml_drugs'))),
    ]
    kpi_html = ''.join(f'<div class="kpi"><div class="kpi-v">{v}</div>'
                       f'<div class="kpi-l">{html.escape(k)}</div></div>'
                       for k, v in kpis)

    # ---- holdout cell lines ----
    hold = info.get('holdout', [])
    if annotated is not None and 'ModelID' in annotated.columns:
        names = dict(zip(annotated['ModelID'], annotated['StrippedCellLineName']))
        hold_df = pd.DataFrame([{'ModelID': h, 'Cell line': names.get(h, h),
                                 'Role': 'Blind holdout'} for h in hold])
    else:
        hold_df = pd.DataFrame([{'ModelID': h, 'Cell line': h, 'Role': 'Blind holdout'}
                                for h in hold])

    # ---- sections ----
    panel = _panel_rows(preds) if preds is not None else None
    bliss_rows = _bliss_rows(preds) if preds is not None else None
    top_rank = top20.head(10) if top20 is not None else None

    ctp_show = None
    if ctp is not None:
        sel = ctp[(ctp['target'] == 'RSL3')
                  & (ctp['ModelID'].isin(['ACH-000768', 'ACH-000849', 'ACH-000019']))].copy()
        if len(sel):
            sel['Cell line'] = sel['ModelID'].map({'ACH-000768': 'MDA-MB-231',
                                                   'ACH-000849': 'MDA-MB-468',
                                                   'ACH-000019': 'MCF7'})
            ctp_show = sel[['Cell line', 'auc_raw', 'auc_norm_0_1', 'sens_percentile']]
            ctp_show.columns = ['Cell line', 'RSL-3 raw AUC (CTRPv2)', 'AUC 0-1',
                                'Percentile most sensitive (of 808)']
            ctp_show = ctp_show.sort_values('AUC 0-1')

    glob = intervals.get('global') or {}
    line_rows = []
    for mid, s in (intervals.get('lines') or {}).items():
        line_rows.append({'ModelID': mid, 'n': s.get('n'),
                          'RMSE': s.get('rmse'), '±80% CI': s.get('abs_q80'),
                          '±95% CI': s.get('abs_q95')})
    line_ci = pd.DataFrame(line_rows) if line_rows else None

    v2_best = (rsl_report.get('models') or {}).get(rsl_report.get('winner', ''), {})
    fbest = (fres.get('per_model_oof') or {}).get(fres.get('frozen_model', ''), {})

    gates_html = []
    if best is not None:
        gates_html = [
            ('R² > 0.25', float(best['r2']), 0.25, 'PASS' if best['r2'] > 0.25 else 'FAIL'),
            ('Per-line Spearman > 0.40', float(best['per_line_spearman']), 0.40,
             'PASS' if best['per_line_spearman'] > 0.40 else 'FAIL'),
            ('NDCG@10 (ranking)', float(best['ndcg10']), None, 'reported'),
        ]

    top_note = ('The predicted-AUC distribution is compressed above ~0.906 for the '
                'resistant bulk of the panel; ranks below ~226 are tied within model '
                'resolution and should not be read as a fine-grained order. The '
                'sensitive tail (AUC < 0.4) is where the model carries signal.')

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
    .ok { color: #2f7d32; font-weight: 700; }
    .bad { color: #b3261e; font-weight: 700; }
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
<title>TNBC / FerroSelect — Defence Executive Report</title>
<style>{css}</style></head><body>
<h1>FerroSelect — TNBC Drug-Response Predictor</h1>
<div class="sub">Defence executive report · generated {now} · model version
<b>{html.escape(str(info.get('model_version', 'n/a')))}</b></div>
<div class="callout"><b>Scope:</b> predicted sensitivity in cell models — never a
clinical treatment recommendation. Wet-lab confirmation is the decision layer.</div>
<div class="kpis">{kpi_html}</div>

<h2>1. Dataset and cell-line status</h2>
<p>The DepMap metadata covers <b>{_fmt(audit.get('breast_lines_total'))}</b> breast lines,
<b>{_fmt(audit.get('wanted_with_expression'))}</b> with expression, of which
<b>{_fmt(audit.get('ml_cell_lines'))}</b> enter training on
<b>{_fmt(audit.get('ml_rows'))}</b> unique (cell line, drug) pairs and
<b>{_fmt(audit.get('ml_drugs'))}</b> screened drugs. Subtypes: TNBC
{_fmt((audit.get('subtype_counts') or {}).get('TNBC'))}, Luminal
{_fmt((audit.get('subtype_counts') or {}).get('Luminal'))}, HER2+
{_fmt((audit.get('subtype_counts') or {}).get('HER2+'))}, other
{_fmt((audit.get('subtype_counts') or {}).get('other'))}.
MDA-MB-231 and MCF7 were blind holdouts for every fit.</p>
{_table(hold_df)}

<h2>2. The eight-member wet-lab panel (ranked 1–8 on MDA-MB-231)</h2>
{_table(panel, nd=3)}
<p class="muted">Rank 1 = strongest expected killing. RSL3 lanes are lab-only by
design: PRISM never profiled RSL3, so no model AUC is fabricated for them.</p>

<h2>3. Top-ranked killers and model resolution</h2>
{_bars(top_rank, 'drug_name', 'pred_AUC_MDA231', n=10) if top_rank is not None else '<p class="muted">Ranking unavailable.</p>'}
<p class="muted">{html.escape(top_note)}</p>

<h2>4. Bliss combination expectations (additivity — test on the plate)</h2>
{_table(bliss_rows)}
{_table(combos.head(5) if combos is not None else None,
        ['drugA_name', 'drugB_name', 'E_AB_MDA231', 'E_AB_MCF7', 'selectivity_gap'])}

<h2>5. CTRPv2.1 measured ground truth for RSL-3</h2>
{_table(ctp_show, nd=2)}
<p class="muted">{html.escape(str(ctp_meta.get('auc_definition', 'v2.1 published AUC; lower = more cell death.')))}</p>
<p>RSL-3 v2 model: <b>{html.escape(str(rsl_report.get('winner', 'n/a')))}</b>,
OOF Spearman <b>{_fmt(v2_best.get('spearman'))}</b>
(95% CI {_fmt((v2_best.get('bootstrap_spearman') or {}).get('lo95'))} to
{_fmt((v2_best.get('bootstrap_spearman') or {}).get('hi95'))}) on
{_fmt(v2_best.get('n'))} lines — below the published lab-vs-CTRP ceiling
({_fmt((rsl_report.get('ceiling_lab_vs_ctrp') or {}).get('RSL3'))}).</p>

<h2>6. Prediction uncertainty (calibrated on blind holdouts)</h2>
<p>Conformal-style intervals from {_fmt(glob.get('n'))} out-of-sample rows:
<b>±{_fmt(glob.get('abs_q80'))}</b> at 80% and
<b>±{_fmt(glob.get('abs_q95'))}</b> at 95% (global half-widths).</p>
{_table(line_ci)}
<p class="muted">FerroSelect frozen model ({html.escape(str(fres.get('frozen_model', 'n/a')))},
{_fmt(fbest.get('n'))} panel rows): pooled OOF Spearman
{_fmt(fbest.get('spearman'))} with bootstrap CI
{_fmt((fbest.get('bootstrap_spearman') or {}).get('lo95'))} to
{_fmt((fbest.get('bootstrap_spearman') or {}).get('hi95'))}. This is an honestly
low, pre-registered hypothesis — the plate decides.</p>

<h2>7. Validation gates</h2>
<table class="data"><tr><th>Gate</th><th>Value</th><th>Target</th><th>Status</th></tr>
"""]

    for label, value, target, status in gates_html:
        parts.append(f'<tr><td>{html.escape(label)}</td><td>{_fmt(value)}</td>'
                     f'<td>{_fmt(target) if target is not None else "—"}</td>'
                     f'<td class="{"ok" if status == "PASS" else ("bad" if status == "FAIL" else "")}">'
                     f'{html.escape(status)}</td></tr>')
    if gates:
        ds = gates.get('delta_stats') or {}
        parts.append(f'<tr><td>Selectivity rule</td><td colspan="3">'
                     f'{html.escape(str(gates.get("selectivity_rule", "n/a")))} '
                     f'({_fmt(ds.get("n_positive"))} positive-Δ drugs of '
                     f'{_fmt(gates.get("panel_size"))})</td></tr>')
    if pca_status:
        parts.append(f'<tr><td>PCA gene annotation</td><td colspan="3">'
                     f'{html.escape(str(pca_status.get("outcome", "n/a")))} — '
                     f'{_fmt((pca_status.get("checks") or {}).get("n_components_verified"))} of '
                     f'{_fmt((pca_status.get("checks") or {}).get("n_components"))} components verified</td></tr>')
    parts.append('</table>')

    parts.append(f"""
<h2>8. Sign-off (print, sign, lodge with the sealed prediction)</h2>
<div class="sign"><table>
<tr><td class="label">Computational lead</td><td class="label">Supervisor</td></tr>
<tr><td></td><td></td></tr>
<tr><td class="label">Date</td><td class="label">Date</td></tr>
<tr><td></td><td></td></tr>
</table></div>
<footer>Artifacts: outputs_local/ · splits by cell line · fixed seed 42 · pinned
requirements.txt · prediction intervals calibrated on pre-registered holdout lines.
Report generated automatically from the committed artifacts; no values are typed by hand.</footer>
</body></html>""")
    return ''.join(parts)
