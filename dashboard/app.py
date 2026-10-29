"""TNBC / FerroSelect Drug Response Predictor — Streamlit dashboard.

Run:  streamlit run dashboard/app.py
Compatible with the pinned Streamlit 1.39 (uses use_container_width).
Reads all artifacts from ../outputs_local (no retraining, no API needed).
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'outputs_local'))
sys.path.insert(0, HERE)

import defense_report          # noqa: E402  (local dashboard modules)
import network_graph           # noqa: E402
import synergy                 # noqa: E402

SENSITIVE_AUC = 0.3
HIGHLIGHT, MUTED = '#D9480F', '#868E96'
POSITIVE_CONTROLS = {'doxorubicin', 'paclitaxel'}
DEFAULT_LINE = 'ACH-000768'             # MDA-MB-231
LINE_NAMES = {'ACH-000768': 'MDA-MB-231', 'ACH-000849': 'MDA-MB-468',
              'ACH-000019': 'MCF7', 'ACH-000049': 'MCF7'}
CSVS = ['mda_mb_231_full_ranking.csv', 'mda_mb_231_top20_killers.csv',
        'wetlab_shortlist.csv', 'combo_rankings.csv', 'leaderboard.csv',
        'baseline_audit.csv']
RANK_COLS = ['rank', 'drug_name', 'pred_AUC_MDA231', 'auc_ci_mda231', 'pred_AUC_MCF7',
             'selectivity_delta', 'sensitive_call', 'selective_call']
AUC_RANGE = (0.2, 0.6)                  # keeps the colour scale informative

st.set_page_config(page_title='FerroSelect / TNBC Drug Response Predictor',
                   layout='wide')

st.markdown("""
<style>
.block-container{padding-top:1.6rem;padding-bottom:2rem;}
[data-testid="stMetricValue"]{font-size:1.9rem;}
[data-testid="stMetric"]{padding-top:.3rem;}
[data-testid="stMetricLabel"]{font-size:.85rem;padding-top:.3rem;line-height:1.35;}
[data-testid="stMetricLabel"] p{margin-bottom:0;}
.stTabs [data-baseweb="tab"]{padding:.45rem .9rem;font-size:.95rem;}
div[data-testid="stDownloadButton"] button{width:100%;}
</style>
""", unsafe_allow_html=True)

# ---------- data ----------
missing = [f for f in CSVS + ['model_info.json']
           if not os.path.exists(os.path.join(DATA, f))]
if missing:
    st.error(f'Missing files in `{DATA}`: ' + ', '.join(f'`{m}`' for m in missing)
             + '. Re-run the export step, then reload this page.')
    st.stop()


@st.cache_data
def _load(name, stamp):
    del stamp                      # mtime bump invalidates the cache when files change
    return pd.read_csv(os.path.join(DATA, name))


def load(name):
    """Read a CSV from outputs_local; cache key includes the file mtime so a
    regenerated file is picked up on the next rerun (stale-cache killer)."""
    path = os.path.join(DATA, name)
    return _load(name, os.path.getmtime(path))


@st.cache_data
def _load_json(name, stamp):
    del stamp
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def load_json(name):
    path = os.path.join(DATA, name)
    return _load_json(name, os.path.getmtime(path))


@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


rank_full = load('mda_mb_231_full_ranking.csv')
top20 = load('mda_mb_231_top20_killers.csv')
shortlist = load('wetlab_shortlist.csv')
combos = load('combo_rankings.csv')
leaderboard = load('leaderboard.csv')
baseline_table = load('baseline_audit.csv')
baseline_audit = (load_json('baseline_audit.json')
                  if os.path.exists(os.path.join(DATA, 'baseline_audit.json')) else None)
expression_audit = (load_json('expression_line_audit.json')
                    if os.path.exists(os.path.join(DATA, 'expression_line_audit.json'))
                    else None)
info = load_json('model_info.json')
intervals = (load_json('prediction_intervals.json')
             if os.path.exists(os.path.join(DATA, 'prediction_intervals.json')) else None)
LINE_IDS = {'MDA-MB-231': 'ACH-000768', 'MDA-MB-468': 'ACH-000849',
            'MCF7': 'ACH-000019'}


def clean_label(s):
    """Strip the winner tick so the UI stays emoji-free."""
    return re.sub(r'[^\x00-\x7F]+', '', str(s)).strip()


leaderboard['label'] = leaderboard['label'].map(clean_label)
lb = leaderboard.set_index('label')
best = lb.sort_values('ndcg10', ascending=False).iloc[0]
champion = clean_label(best.name)

# ---------- display helpers ----------
SENS_MAP = {'SENSITIVE (AUC<0.3)': 'Sensitive', 'sensitive': 'Sensitive',
            'resistant/intermediate': 'Resistant', 'resistant': 'Resistant',
            'RESISTANT (AUC>=0.3)': 'Resistant'}
SEL_MAP = {'selective': 'Predicted selective (unvalidated)',
           'non-selective': 'Not predicted selective',
           'tnbc-selective': 'Predicted selective (unvalidated)'}
PENDING = 'Pending annotation'


@st.cache_data
def selectivity_rule():
    path = os.path.join(DATA, 'selectivity_rule.json')
    return json.load(open(path)) if os.path.exists(path) else None


def selectivity_caption():
    r = selectivity_rule()
    if not r:
        return ('**Predicted selective (unvalidated)** = the model predicts stronger '
                'killing of MDA-MB-231 than MCF7. This is not a measured selectivity call.')
    ds = r['delta_stats']
    null = ((baseline_audit or {}).get('selectivity') or {}).get('all_shared_drugs') or {}
    restricted = ((baseline_audit or {}).get('selectivity') or {}).get(
        'restricted_measured_auc_below_0_5_either') or {}
    null_text = ''
    if null.get('spearman', {}).get('statistic') is not None:
        rs = restricted.get('spearman', {}).get('statistic')
        rs_text = f'{rs:.3f}' if isinstance(rs, (int, float)) else '—'
        null_text = (
            f"  \n**Post-hoc, exploratory null:** predicted Δ is uncorrelated with "
            f"measured Δ on the {null['n']:,} drugs shared by MDA-MB-231 and MCF7 "
            f"(Pearson {null['pearson']['statistic']:.3f}, "
            f"Spearman {null['spearman']['statistic']:.3f}, sign agreement "
            f"{null['sign_agreement']:.1%}). The restricted check (measured AUC < 0.5 "
            f"in either line, n={restricted.get('n', '—')}) gives Spearman "
            f"{rs_text}. The badge is a hypothesis to test by SI, not a finding.")
    return (f"**Predicted selective (unvalidated)** = {r['selectivity_rule']}  \n"
            f"Only {ds['n_positive']:,} of {r['panel_size']:,} drugs have a positive Δ "
            f"(the old fixed cut-off Δ > 0.05 was unreachable: max Δ = {ds['max']:.3f}), "
            f"so the relative rule is what carries information here. "
            f"**Sensitive** = {r['sensitivity_rule']}." + null_text)


def pretty(df):
    """Round the numbers a bench scientist has to read, shorten the call columns."""
    d = df.copy()
    for c in d.columns:
        if c.startswith(('pred_AUC', 'E_AB', 'selectivity')):
            d[c] = d[c].astype(float).round(3)
    if 'sensitive_call' in d.columns:
        d['sensitive_call'] = d['sensitive_call'].astype(str).map(
            lambda v: SENS_MAP.get(v.strip(), SENS_MAP.get(v.strip().lower(), v.strip())))
    if 'selective_call' in d.columns:
        d['selective_call'] = d['selective_call'].astype(str).map(
            lambda v: SEL_MAP.get(v.strip().lower(), v.strip()))
    return d


def ci_halfwidth(line_id=None, drug_id=None, level=80):
    """Half-width of the calibrated prediction interval (per-drug → per-line → global).

    The intervals are conformal-style empirical quantiles of the residuals on the
    seven blind holdout lines (`src/models/prediction_intervals.py`). They are
    marginal, not conditional guarantees.
    """
    if not intervals:
        return float('nan')
    key = 'abs_q80' if int(level) == 80 else 'abs_q95'
    if drug_id is not None and pd.notna(drug_id):
        v = (intervals.get('drugs') or {}).get(str(drug_id), {}).get(key)
        if v is not None and np.isfinite(v):
            return float(v)
    if line_id is not None and pd.notna(line_id):
        v = (intervals.get('lines') or {}).get(str(line_id), {}).get(key)
        if v is not None and np.isfinite(v):
            return float(v)
    v = (intervals.get('global') or {}).get(key)
    return float(v) if v is not None and np.isfinite(v) else float('nan')


def ci_label(auc, line_id=None, drug_id=None, level=80):
    """Render `AUC ± half-width` (the required presentation) without inventing a number."""
    if auc is None or pd.isna(auc):
        return '—'
    hw = ci_halfwidth(line_id, drug_id, level)
    if not np.isfinite(hw):
        return f'{float(auc):.3f}'
    return f'{float(auc):.3f} ± {hw:.3f}'


def attach_ci(df, line_id, drug_col='drug_id', level=80):
    """Add the display column (`auc_ci_mda231`) and the numeric error array."""
    d = df.copy()
    if drug_col in d.columns:
        hw = d[drug_col].map(lambda v: ci_halfwidth(line_id, v, level))
    else:
        hw = pd.Series(ci_halfwidth(line_id, None, level), index=d.index)
    d['auc_hw80'] = hw.astype(float)
    d['auc_ci_mda231'] = [ci_label(v, None, None) if not np.isfinite(h)
                          else f'{float(a):.3f} ± {float(h):.3f}'
                          for a, h in zip(d.get('pred_AUC_MDA231', pd.Series(np.nan, index=d.index)), hw)]
    return d


rank_full = attach_ci(rank_full, DEFAULT_LINE)
top20 = attach_ci(top20, DEFAULT_LINE)
shortlist = attach_ci(shortlist, DEFAULT_LINE)


def badge_styler(df, cols, palette):
    """Colour a text column with cell background so the eye scans it instantly."""
    rules = {c: palette for c in cols}
    out = df.style
    for c in df.columns:
        if c in rules:
            out = out.map(lambda v: _badge_css(v, rules[c]), subset=[c])
    return out


def _badge_css(v, palette):
    key = str(v).strip().lower()
    for token, css in palette.items():
        if key.startswith(token):
            return css
    return ''


GREEN = 'background-color:#2f7d32;color:#ffffff;font-weight:600'
RED = 'background-color:#b3261e;color:#ffffff;font-weight:600'
GREY = 'background-color:#5f6368;color:#ffffff;font-weight:600'
SENS_PALETTE = {'sensitive': GREEN, 'resistant': RED}
PENDING_PALETTE = {'pending': GREY, '?': GREY}

RANK_COL_CFG = {
    'rank': st.column_config.NumberColumn('Rank', format='%d', width='small'),
    'drug_name': st.column_config.TextColumn('Drug'),
    'pred_AUC_MDA231': st.column_config.ProgressColumn(
        'MDA-MB-231 AUC', format='%.3f', min_value=0.0, max_value=1.0,
        help='Predicted AUC. The bar is only a visual aid — shorter bar = stronger killing.'),
    'auc_ci_mda231': st.column_config.TextColumn(
        'Predicted AUC ± 80% CI',
        help='Conformal 80% interval calibrated on the seven blind holdout lines '
             '(per-line half-width; falls back to the global ±0.104).'),
    'pred_AUC_MCF7': st.column_config.NumberColumn(
        'MCF7 AUC', format='%.3f', help='Predicted AUC in the luminal control.'),
    'selectivity_delta': st.column_config.NumberColumn(
        'Selectivity Δ', format='%.3f',
        help='MCF7 AUC minus MDA-MB-231 AUC. Positive = hits TNBC harder than control.'),
    'sensitive_call': st.column_config.TextColumn('Verdict', help='Cut-off at AUC 0.30'),
    'selective_call': st.column_config.TextColumn('Selectivity call (unvalidated)'),
    'role': st.column_config.TextColumn('Role'),
}


def rank_table(df):
    df = pretty(df)
    cfg = {c: RANK_COL_CFG[c] for c in df.columns if c in RANK_COL_CFG}
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=cfg)


def table_with_download(df, filename, key, cfg=None, styler=False):
    """Table with the export button in the top-right corner, not buried below it."""
    left, right = st.columns([4, 1])
    with left:
        if styler:
            st.dataframe(df, hide_index=True, use_container_width=True, column_config=cfg or {})
        else:
            st.dataframe(df, hide_index=True, use_container_width=True, column_config=cfg or {})
    with right:
        st.download_button('Download CSV', to_csv(df), filename, 'text/csv', key=key)


# ---------- charts (plotly: the Altair/Vega spec was rejected by the browser) ----------
def bar_chart(df):
    """Horizontal bars, rank order, full drug names, diverging colour on selectivity Δ."""
    d = pretty(df)
    fig = px.bar(d, x='pred_AUC_MDA231', y='drug_name', orientation='h',
                 color='selectivity_delta',
                 color_continuous_scale='RdYlGn_r', color_continuous_midpoint=0,
                 range_color=[float(d['selectivity_delta'].min()) - 0.01,
                              float(d['selectivity_delta'].max()) + 0.01],
                 hover_data={'pred_AUC_MDA231': ':.3f', 'pred_AUC_MCF7': ':.3f',
                             'selectivity_delta': ':.3f', 'drug_name': False},
                 labels={'pred_AUC_MDA231': 'Predicted AUC (lower = stronger kill)',
                         'drug_name': None, 'selectivity_delta': 'Selectivity Δ'})
    fig.update_layout(height=max(260, 30 * len(d)), margin=dict(l=230, t=10, b=40),
                      yaxis={'automargin': True}, coloraxis_colorbar_title='Δ')
    # 80% conformal error bars: the bar is a point estimate, the whisker is the
    # honest uncertainty calibrated on blind holdout lines.
    hw = None
    if 'auc_hw80' in d.columns:
        hw = pd.to_numeric(d['auc_hw80'], errors='coerce')
    elif 'auc_ci_mda231' in d.columns:
        hw = pd.to_numeric(d['auc_ci_mda231'].astype(str).str.extract(
            r'±\s*([0-9.]+)')[0], errors='coerce')
    if hw is not None and hw.notna().any():
        fig.update_traces(error_x=dict(type='data', array=hw.fillna(0).tolist(),
                                       visible=True, thickness=1.3, width=3,
                                       color='#343A40'))
    return fig


def selectivity_chart(df, n_top):
    """Scatter vs the y=x diagonal, sensitivity cut-off, labels for the top N."""
    d = df.dropna(subset=['pred_AUC_MDA231', 'pred_AUC_MCF7', 'selectivity_delta']).copy()
    d = pretty(d)
    d['label'] = np.where(d['rank'] <= n_top, d['drug_name'], '')
    lo = float(min(d['pred_AUC_MCF7'].min(), d['pred_AUC_MDA231'].min()))
    hi = float(max(d['pred_AUC_MCF7'].max(), d['pred_AUC_MDA231'].max()))
    pad = 0.02 * (hi - lo) or 0.01

    fig = px.scatter(d, x='pred_AUC_MCF7', y='pred_AUC_MDA231', text='label',
                     color='selectivity_delta',
                     color_continuous_scale='RdYlGn_r', color_continuous_midpoint=0,
                     hover_name='drug_name', hover_data={
                         'pred_AUC_MDA231': ':.3f', 'pred_AUC_MCF7': ':.3f',
                         'selectivity_delta': ':.3f', 'label': False},
                     labels={'pred_AUC_MCF7': 'Predicted AUC, MCF7 (control)',
                             'pred_AUC_MDA231': 'Predicted AUC, MDA-MB-231 (TNBC)',
                             'selectivity_delta': 'Selectivity Δ'})
    fig.update_traces(textposition='top center', textfont=dict(size=10, color=HIGHLIGHT),
                      marker=dict(size=9, opacity=0.75, line=dict(width=0.5, color='#ffffff')))
    fig.add_shape(type='line', x0=lo - pad, y0=lo - pad, x1=hi + pad, y1=hi + pad,
                  line=dict(dash='dash', color=MUTED, width=1))
    fig.add_hline(y=SENSITIVE_AUC, line=dict(dash='dot', color=MUTED, width=1))
    fig.add_vline(x=SENSITIVE_AUC, line=dict(dash='dot', color=MUTED, width=1))
    fig.update_xaxes(range=[lo - pad, hi + pad])
    fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_layout(height=520, margin=dict(t=10, b=40),
                      coloraxis_colorbar_title='Δ')
    return fig


def rescue_bubble(df):
    """Bubble chart: killers (x) vs rescue enrichment (y); size = re-expressed
    proteins, colour = rescue control. The one-glance picture of rescue power.

    NAC is excluded: the archived NAC rows are a Solifenacin-Succinate signature
    produced by a substring match on "SolifeNACin" (see the provenance audit).
    """
    d = df[df['rescue'].map(lambda v: str(v).upper()) != 'N-ACETYLCYSTEINE'].copy()
    d['rescue'] = d['rescue'].replace({'DEFEROXAMINE': 'DFO'})
    d['p_text'] = d['reversal_p'].map(
        lambda v: '< 1e-16' if not v or v == 0 else f'{v:.1g}')
    fig = px.scatter(d, x='killer', y='reversal_enrich',
                     size='reversal_hits', size_max=46,
                     color='rescue', color_discrete_map={'DFO': '#1c7ed6'},
                     hover_name='killer',
                     hover_data={'reversal_enrich': ':.2f', 'reversal_hits': True,
                                 'p_text': True, 'rescue': True,
                                 'signature_source': True, 'killer': False,
                                 'reversal_p': False},
                     category_orders={'killer': ['ERASTIN', 'SULFASALAZINE', 'RSL3',
                                                 'SIMVASTATIN']},
                     labels={'killer': 'Killer drug', 'rescue': 'Rescue control',
                             'reversal_enrich': 'Rescue enrichment (x)',
                             'reversal_hits': 'Re-expressed proteins',
                             'p_text': 'p (hypergeom)',
                             'signature_source': 'Killer signature source'})
    y_hi = float(d['reversal_enrich'].max()) * 1.18 or 1
    fig.add_hline(y=1, line=dict(dash='dash', color=MUTED, width=1),
                  annotation_text='1x = no enrichment (no reversal signal)',
                  annotation_position='bottom right',
                  annotation_font=dict(color=MUTED, size=10))
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color='#ffffff')))
    fig.update_layout(height=430, margin=dict(t=10, b=40), yaxis=dict(range=[0, y_hi]))
    return fig


def array_priority_bar(df):
    """Bar chart: the rescue-competent array candidates, split by protein location."""
    loc = {'membrane': 'Membrane', 'secreted': 'Secreted', 'intracellular': 'Intracellular'}
    counts = df.drop_duplicates('Gene')['Array-friendly'].value_counts().reindex(
        ['membrane', 'secreted', 'intracellular']).fillna(0).astype(int)
    counts.index = counts.index.map(loc)
    fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                 color_discrete_map={'Membrane': '#2f7d32', 'Secreted': '#2f7d32',
                                     'Intracellular': '#5f6368'},
                 labels={'x': 'Where the protein sits', 'y': 'Unique proteins',
                         'color': 'Array-ready'})
    fig.update_traces(texttemplate='%{y}', textposition='outside',
                      marker_line_width=0)
    fig.update_layout(height=380, margin=dict(t=20, b=40), showlegend=False,
                      yaxis_title='Unique rescue-competent proteins',
                      xaxis_title='Protein location')
    return fig


PANEL_RANK_ORDER = [
    (1, 'ERASTIN + SIMVASTATIN', 'Combo', 'combo'),
    (2, 'SULFASALAZINE + SIMVASTATIN', 'Combo (sealed nomination)', 'combo'),
    (3, 'RSL3 + SIMVASTATIN', 'Combo', 'rsl3combo'),
    (4, 'ERASTIN', 'Killer (Type I)', 'ERASTIN'),
    (5, 'SULFASALAZINE', 'Killer (Type I)', 'SULFASALAZINE'),
    (6, 'RSL3', 'Killer (Type II, GPX4)', 'rsl3'),
    (7, 'N-ACETYLCYSTEINE (NAC)', 'Rescue control', 'NAC'),
    (8, 'DEFEROXAMINE (DFO)', 'Rescue control', 'DFO'),
]
ROLE_COLORS = {'Killer': '#b3261e', 'Rescue control': '#1c7ed6', 'Combo': '#d9480f'}


def _bliss(e, a, b):
    """Bliss expected effect from two single-agent effects; NaN if either is missing."""
    if a not in e.index or b not in e.index:
        return np.nan
    return float(e[a] + e[b] - e[a] * e[b])


def panel_ranking_table(preds):
    """The agreed 8-member panel ranked 1-8 on MDA-MB-231, sealed numbers only."""
    p231 = preds[preds['line'] == 'MDA-MB-231'].set_index('drug')
    auc = p231['pred_auc'].astype(float)
    nrows = p231['n_train_rows'].to_dict()
    e = 1.0 - auc

    supp = {}
    if os.path.exists(os.path.join(DATA, 'offtarget_panel_map.csv')):
        m = load('offtarget_panel_map.csv')
        m = m[m['line'] == DEFAULT_LINE]
        def hits(drug, direction):
            return int(m[(m['drug'] == drug) & (m['direction'] == direction)]['hit_genes'].sum())
        supp['SIMVASTATIN_down'] = hits('SIMVASTATIN', 'Down')
        supp['DFO_up'] = hits('DEFEROXAMINE', 'Up')
    rsl3_n = None
    if os.path.exists(os.path.join(DATA, 'rsl3_mini_signature_meta.json')):
        rsl3_n = load_json('rsl3_mini_signature_meta.json').get('n_down_expressed_mda231')

    combined = None
    if rsl3_n and supp.get('SIMVASTATIN_down'):
        combined = rsl3_n + supp['SIMVASTATIN_down']

    def value(key):
        if key == 'ERASTIN':
            return float(auc.get('ERASTIN', np.nan)), f'{nrows.get("ERASTIN")} PRISM training rows (LOW confidence)'
        if key == 'SULFASALAZINE':
            return float(auc.get('SULFASALAZINE', np.nan)), f'{nrows.get("SULFASALAZINE")} PRISM training rows (LOW confidence)'
        if key == 'combo':
            return np.nan, 'Lab-only: PRISM has no AUC for RSL3, so no Bliss number exists'
        if key == 'rsl3combo':
            note = ('Lab-only (no PRISM AUC for RSL3); together the two signatures produce '
                    f'{combined} suppression events in MDA-MB-231 '
                    f'({rsl3_n} RSL3 + {supp["SIMVASTATIN_down"]} simvastatin; '
                    f'{combined - 2} unique proteins, 2 shared)') if combined \
                else 'Lab-only (no PRISM AUC for RSL3)'
            return np.nan, note
        if key == 'rsl3':
            note = (f'Type II GPX4 killer, lab-only; {rsl3_n} expressed proteins suppressed in '
                    f'MDA-MB-231; dose 0.01-5 uM') if rsl3_n else 'Type II GPX4 killer, lab-only'
            return np.nan, note
        if key == 'NAC':
            # The legacy L1000 matcher resolved "NAC" inside "SolifeNACin"; the archived
            # 166 "NAC up" genes are Solifenacin. Never present that count as NAC evidence.
            note = ('Rescue control; signature WITHHELD — the legacy L1000 match resolved '
                    'to Solifenacin, not N-acetylcysteine (provenance audit).')
            return np.nan, note
        if key == 'DFO':
            note = (f'Rescue control; re-expresses {supp["DFO_up"]} proteins in MDA-MB-231 '
                    f'(L1000 Up signature)') if supp.get('DFO_up') else 'Rescue control'
            return np.nan, note
        return np.nan, ''

    rows = []
    for rank, member, role, key in PANEL_RANK_ORDER:
        if role.startswith('Combo') and key == 'combo':
            killer = member.split(' + ')[0]
            auc_val = (1 - _bliss(e, killer, 'SIMVASTATIN')) if killer in e.index else np.nan
            note = ('Sealed nomination' if 'nomination' in role
                    else 'Bliss expectation from the sealed single-agent AUCs')
            if not np.isnan(auc_val):
                note += (f'; E_AB = {1 - auc_val:.3f}; Bliss arithmetic, not synergy — '
                         'needs the checkerboard test')
        else:
            auc_val, note = value(key)
        rows.append({'Rank': rank, 'Panel member': member, 'Role': role,
                     'Predicted AUC (MDA-MB-231)': auc_val,
                     'AUC 80% CI (±)': (ci_halfwidth(DEFAULT_LINE) if not pd.isna(auc_val)
                                        else np.nan),
                     'Evidence / note': note})
    d = pd.DataFrame(rows)
    d['Predicted AUC (MDA-MB-231)'] = d['Predicted AUC (MDA-MB-231)'].astype(float).round(3)
    d['AUC 80% CI (±)'] = d['AUC 80% CI (±)'].astype(float).round(3)
    return d


def panel_ranking_chart(d):
    """Ranked 1-8 bars: predicted AUC on MDA-MB-231, colour by the wet-lab role."""
    def bucket(role):
        if role.startswith('Rescue'):
            return 'Rescue control'
        return 'Combo' if role.startswith('Combo') else 'Killer'
    dd = d.assign(role=[bucket(r) for r in d['Role']]).iloc[::-1]
    fig = px.bar(dd, x='Predicted AUC (MDA-MB-231)', y='Panel member', orientation='h',
                 color='role', color_discrete_map=ROLE_COLORS,
                 category_orders={'Panel member': d['Panel member'].tolist()[::-1]},
                 labels={'Predicted AUC (MDA-MB-231)':
                         'Predicted AUC on MDA-MB-231 (lower = stronger expected killing)',
                         'role': 'Role'})
    fig.update_traces(texttemplate='%{x:.3f}', textposition='outside',
                      marker_line_width=0)
    if 'AUC 80% CI (±)' in dd.columns:
        hw = pd.to_numeric(dd['AUC 80% CI (±)'], errors='coerce')
        fig.update_traces(error_x=dict(type='data',
                                       array=[None if pd.isna(v) else float(v) for v in hw],
                                       visible=True, thickness=1.2, width=3,
                                       color='#343A40'))
    for _, r in d.iterrows():
        if pd.isna(r['Predicted AUC (MDA-MB-231)']):
            fig.add_annotation(x=0.005, y=r['Panel member'], text='no AUC — lab-only lane',
                               showarrow=False, xanchor='left',
                               font=dict(color=MUTED, size=10))
    top = float(d['Predicted AUC (MDA-MB-231)'].max()) * 1.18 or 1.0
    fig.update_layout(height=430, margin=dict(t=10, b=40, l=215),
                      xaxis=dict(range=[0, top]), yaxis_title=None)
    return fig


def combo_bliss_chart(preds):
    """The three Simvastatin pairs: Bliss expected effect, grouped by cell line."""
    rows = []
    for line in ['MDA-MB-231', 'MDA-MB-468', 'MCF7']:
        p = preds[preds['line'] == line].set_index('drug')['pred_auc'].astype(float)
        e = 1.0 - p
        for killer in ['ERASTIN', 'SULFASALAZINE', 'RSL3']:
            combo = f'{killer} + SIMVASTATIN'
            rows.append({'combo': combo, 'line': line,
                         'E_AB': _bliss(e, killer, 'SIMVASTATIN')
                         if killer in e.index else np.nan})
    d = pd.DataFrame(rows)
    order = ['RSL3 + SIMVASTATIN', 'SULFASALAZINE + SIMVASTATIN',
             'ERASTIN + SIMVASTATIN']
    fig = px.bar(d, x='E_AB', y='combo', orientation='h', barmode='group',
                 color='line',
                 color_discrete_map={'MDA-MB-231': '#b3261e',
                                     'MDA-MB-468': '#f59f00', 'MCF7': '#495057'},
                 category_orders={'combo': order},
                 labels={'E_AB': 'Bliss expected effect E_AB', 'line': 'Line'})
    fig.update_traces(texttemplate='%{x:.3f}', textposition='outside',
                      marker_line_width=0)
    fig.add_annotation(x=0.002, y='RSL3 + SIMVASTATIN',
                       text='not computable - no PRISM AUC for RSL3',
                       showarrow=False, xanchor='left',
                       font=dict(color=MUTED, size=10))
    fig.update_layout(height=340, margin=dict(t=10, b=40, l=210),
                      xaxis=dict(range=[0, 0.35]))
    return fig


# ---------- live model (loaded only on request) ----------
@st.cache_resource(show_spinner='Loading CatBoost model…')
def load_live():
    import pyarrow.parquet as pq
    from catboost import CatBoostRegressor
    cols = pq.read_schema(os.path.join(DATA, 'tnbc_ml_dataset.parquet')).names
    feat = [c for c in cols if c.startswith('pca_')
            or (c.startswith('drug_') and c not in ('drug_id', 'drug_name'))]
    pos = {c: i for i, c in enumerate(feat)}
    pca = pd.read_parquet(os.path.join(DATA, 'expression_pca128_breast.parquet'))
    pca = pca.drop_duplicates(subset=['ModelID'], keep='first').reset_index(drop=True)
    model = CatBoostRegressor()
    model.load_model(os.path.join(DATA, 'catboost_tnbc_v1.cbm'))
    # Drugs the model actually learned: a drug column with zero importance was never
    # split on, so scoring it returns the drug-free baseline - not a prediction.
    imp = pd.Series(model.get_feature_importance(), index=feat)
    scored = {c[len('drug_'):] for c, v in imp.items() if c.startswith('drug_') and v > 0}
    return feat, pos, pca, model, scored


def live_tab():
    st.subheader('Live predictor: any profiled line × any screened drug')
    st.write('Predicts on demand with the trained CatBoost model, for questions beyond '
             'the precomputed lines (for example BT-549). Values for MDA-MB-231 and '
             'MCF7 match the rankings tab.')
    if not st.toggle('Load the live model',
                     help='Loads CatBoost and the expression table. Takes a few seconds the first time.'):
        return
    try:
        feat, pos, pca_live, model, scored_drugs = load_live()
    except Exception as e:
        st.warning('Could not load the live model. The precomputed tabs still work.')
        st.caption(f'Details: {type(e).__name__}: {e}')
        return

    pca_cols = [c for c in feat if c.startswith('pca_')]
    line_ids = sorted(pca_live['ModelID'].unique().tolist())
    drug_ids = sorted(rank_full['drug_id'].astype(str).unique().tolist())
    name_of = dict(zip(rank_full['drug_id'].astype(str), rank_full['drug_name']))
    st.caption(f'Model coverage: the champion model split on {len(scored_drugs):,} of the '
               f'{len(drug_ids):,} screened drugs. A drug outside that set carries no '
               f'learned signal — its column was never used in training — so scoring it '
               f'returns the drug-free baseline for that cell line, not a drug-specific '
               f'prediction. Such drugs are marked in the dropdown.')

    def _drug_option(d):
        mark = '' if d in scored_drugs else '  · no model signal'
        return f'{name_of.get(d, d)} ({d}){mark}'

    ann_path = os.path.join(DATA, 'breast_lines_annotated.csv')
    if os.path.exists(ann_path):
        ann = load('breast_lines_annotated.csv')
        cell_name = dict(zip(ann['ModelID'], ann['StrippedCellLineName']))
    else:
        cell_name = {}
    pretty_lines = LINE_NAMES

    with st.form('live_form'):
        c1, c2 = st.columns(2)
        sel_line = c1.selectbox(
            'Cell line', line_ids,
            format_func=lambda mid: f"{pretty_lines.get(mid) or cell_name.get(mid, mid)} ({mid})",
            index=line_ids.index(DEFAULT_LINE) if DEFAULT_LINE in line_ids else 0)
        sel_drug = c2.selectbox('Drug', drug_ids, format_func=_drug_option)
        go = st.form_submit_button('Predict sensitivity', type='primary')

    if go:
        if f'drug_{sel_drug}' not in pos:
            st.error(f'Drug {sel_drug} was not in the training features, so the model cannot score it.')
            return
        row = pca_live.loc[pca_live['ModelID'] == sel_line, pca_cols]
        x = np.zeros((1, len(feat)), dtype=np.float32)
        x[0, [pos[c] for c in pca_cols]] = row.to_numpy(dtype=np.float32)[0]
        x[0, pos[f'drug_{sel_drug}']] = 1.0
        auc = float(np.clip(model.predict(x)[0], 0, 1))
        hw = ci_halfwidth(sel_line, sel_drug, 80)
        shown = f'{auc:.3f} ± {hw:.3f}' if np.isfinite(hw) else f'{auc:.3f}'
        m1, m2, m3 = st.columns(3)
        m1.metric('Predicted AUC ± 80% CI', shown,
                  help='Conformal interval calibrated on the blind holdout lines. '
                       'Per-drug intervals are used when available, otherwise the line/global fallback.')
        st.caption(f'80% prediction interval half-width: ±{hw:.3f}. '
                   f'95% half-width: ±{ci_halfwidth(sel_line, sel_drug, 95):.3f}.'
                   if np.isfinite(hw) else 'No calibrated interval available for this line.')
        if auc < SENSITIVE_AUC:
            m2.success(f'Sensitive (AUC < {SENSITIVE_AUC})')
        else:
            m2.info('Resistant or intermediate')
        name = name_of.get(sel_drug, sel_drug)
        if sel_drug not in scored_drugs:
            x0 = np.zeros((1, len(feat)), dtype=np.float32)
            x0[0, [pos[c] for c in pca_cols]] = row.to_numpy(dtype=np.float32)[0]
            base = float(np.clip(model.predict(x0)[0], 0, 1))
            m3.warning('no learned signal')
            st.warning(f'{name} has no learned signal in the champion model, so this number is '
                       f'the drug-free baseline for this cell line (AUC {base:.3f}), not a '
                       f'prediction about {name}. For the panel agents use the FerroSelect '
                       f'panel predictions (the Lasso/Ridge/RF track scores them separately).')
        else:
            ref = rank_full.loc[rank_full['drug_id'].astype(str) == sel_drug, 'pred_AUC_MDA231']
            m3.success('scored by the model')
            if len(ref):
                st.success(f'Reproduces the sealed rankings tab for {name} on '
                           f'MDA-MB-231: AUC {float(ref.iloc[0]):.3f} (Δ {auc - float(ref.iloc[0]):+.3f}).')


# ---------- new feature tabs: synergy, network, defence report ----------
SYNERGY_PAIRS = {
    'ERASTIN + SIMVASTATIN': ('ERASTIN', 'SIMVASTATIN'),
    'RSL3 + SIMVASTATIN': ('RSL3', 'SIMVASTATIN'),
    'SULFASALAZINE + SIMVASTATIN': ('SULFASALAZINE', 'SIMVASTATIN'),
    'ERASTIN + N-ACETYLCYSTEINE (rescue test)': ('ERASTIN', 'N-ACETYLCYSTEINE'),
}
DEFAULT_DOSES = '0, 0.1, 0.3, 1, 3, 10, 30, 100'


def _panel_predictions():
    path = os.path.join(DATA, 'ferroselect_panel_predictions.csv')
    return load('ferroselect_panel_predictions.csv') if os.path.exists(path) else None


def synergy_tab():
    """Interactive checkerboard: paste plate readings → Bliss / ZIP / Loewe scores."""
    st.subheader('Interactive synergy checkerboard (Bliss, ZIP, Loewe)')
    st.write('Enter the measured viability matrix for a two-drug plate (1.0 = fully alive, '
             '0.0 = dead). Row 1 / column 1 must be the zero dose of drug A / drug B; the '
             'calculator reads the single-agent curves from them and computes the synergy '
             'surface immediately.')
    st.caption('The grid starts as a **model-additive placeholder** (V_AB = V_A × V_B), not '
               'measured data — by construction its Bliss excess is zero. Replace the cells '
               'with plate readings; the scores update on every edit.')

    preds = _panel_predictions()
    c1, c2, c3 = st.columns([2, 1, 1])
    pair = c1.selectbox('Drug pair', list(SYNERGY_PAIRS), key='syn_pair')
    line_name = c2.selectbox('Cell line', list(LINE_IDS), key='syn_line')
    unit = c3.selectbox('Concentration unit', ['µM', 'nM', 'mM'], key='syn_unit')
    drug_a, drug_b = SYNERGY_PAIRS[pair]

    d1, d2 = st.columns(2)
    a_text = d1.text_input(f'{drug_a} concentrations ({unit})', DEFAULT_DOSES,
                           key='syn_doses_a')
    b_text = d2.text_input(f'{drug_b} concentrations ({unit})', DEFAULT_DOSES,
                           key='syn_doses_b')
    try:
        conc_a = synergy.parse_conc_text(a_text)
        conc_b = synergy.parse_conc_text(b_text)
    except ValueError as e:
        st.error(f'Concentration series: {e}')
        return

    ref = synergy.model_reference_effect(preds, line_name, drug_a, drug_b) if preds is not None else None
    auc_a = ref['auc_a'] if ref else None
    auc_b = ref['auc_b'] if ref else None
    placeholder = synergy.placeholder_matrix(conc_a, conc_b, auc_a, auc_b)

    demo = st.toggle(
        'Load an illustrative synergy demo (not measured data)',
        value=False,
        help='Fills the grid with a synthetic matrix that is additive on the single-agent '
             'wells and strongly synergistic in the combination block. Use it to preview the '
             'calculator; clear the toggle and paste real plate readings for science.',
        key=f'syn_demo_{pair}_{line_name}')
    mat = placeholder
    if demo:
        arr = placeholder.to_numpy(float).copy()
        if arr.shape[0] > 1 and arr.shape[1] > 1:
            arr[1:, 1:] = np.clip(arr[1:, 1:] * 0.55, 0.0, 1.0)
        mat = pd.DataFrame(arr, index=placeholder.index, columns=placeholder.columns)

    show = mat.copy()
    show.index = [f'{v:g}' for v in conc_a]
    show.columns = [f'{v:g}' for v in conc_b]
    show = show.reset_index()
    first_col = f'Drug A \\ Drug B ({unit})'
    show = show.rename(columns={show.columns[0]: first_col})
    col_cfg = {first_col: st.column_config.TextColumn(first_col, disabled=True, width='small')}
    for col in show.columns[1:]:
        col_cfg[col] = st.column_config.NumberColumn(col, min_value=0.0, max_value=2.0,
                                                     step=0.01, format='%.3f')
    st.write('**Plate readings (viability, editable)**')
    edited = st.data_editor(show, hide_index=True, use_container_width=True,
                            column_config=col_cfg,
                            key=f'syn_editor_{pair}_{line_name}_{len(conc_a)}x{len(conc_b)}_{int(demo)}')
    try:
        viab = edited.drop(columns=[first_col]).apply(pd.to_numeric, errors='coerce').to_numpy(float)
    except Exception as e:
        st.error(f'Could not read the matrix: {e}')
        return
    if viab.shape != (len(conc_a), len(conc_b)):
        st.error('The edited matrix no longer matches the concentration series.')
        return
    if not np.isfinite(viab).all():
        st.error('Some cells are empty or non-numeric — fill every cell before scoring.')
        return
    if (viab < 0).any():
        st.error('Viability cannot be negative.')
        return

    placeholder_arr = placeholder.to_numpy(float)
    untouched = (not demo) and np.allclose(viab, placeholder_arr, atol=1e-9)
    if untouched:
        st.info('This is still the **model-additive placeholder** (V_AB = V_A × V_B). '
                'Its observed matrix is shown below, but Bliss/ZIP/Loewe scores are '
                'suppressed until you replace the cells with plate readings or switch on the '
                'illustrative demo. That prevents a placeholder from ever looking like synergy.')
        st.plotly_chart(synergy.observed_heatmap(viab, conc_a, conc_b, unit),
                        use_container_width=True, config={'displaylogo': False})
        st.caption('The grid is editable above. One changed cell is enough to activate the '
                   'three scores; press Ctrl+Enter or click outside the cell to commit the edit.')
        return

    res = synergy.analyse(viab, conc_a, conc_b)
    bliss, zipr, loewe = res['bliss'], res['zip'], res['loewe']
    observed_top = float(np.clip(1.0 - viab[-1, -1], 0.0, 1.0))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Observed killing (top doses)', f'{observed_top:.3f}',
              help='1 − viability at the highest concentration of both drugs.')
    k2.metric('Mean Bliss excess', f"{bliss['summary']['mean_excess']:+.3f}",
              help='Positive = more killing than additive expectation.')
    k3.metric('Synergistic cells (Bliss > +0.10)',
              f"{bliss['summary']['n_synergistic']}/{bliss['summary']['n_cells']}")
    med_ci = loewe['summary']['median_ci']
    k4.metric('Median Loewe CI', '—' if not np.isfinite(med_ci) else f'{med_ci:.2f}',
              help='Chou-Talalay combination index: < 1 synergy, 1 additive, > 1 antagonism.')

    st.write('**Observed viability** and **Bliss excess** (red = synergy, blue = antagonism)')
    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(synergy.observed_heatmap(viab, conc_a, conc_b, unit),
                        use_container_width=True, config={'displaylogo': False})
    with h2:
        st.plotly_chart(synergy.bliss_heatmap(viab, conc_a, conc_b, unit),
                        use_container_width=True, config={'displaylogo': False})
    st.write('**ZIP excess** (4PL-smoothed single-agent expectation) and '
             '**Loewe / Chou-Talalay CI**')
    h3, h4 = st.columns(2)
    with h3:
        st.plotly_chart(synergy.zip_heatmap(viab, conc_a, conc_b, unit),
                        use_container_width=True, config={'displaylogo': False})
    with h4:
        st.plotly_chart(synergy.loewe_heatmap(viab, conc_a, conc_b, unit),
                        use_container_width=True, config={'displaylogo': False})

    st.write('**Score summary**')
    st.dataframe(pd.DataFrame([
        {'Method': 'Bliss independence',
         'Synergy statistic': 'mean excess (observed − expected)',
         'Value': round(bliss['summary']['mean_excess'], 3),
         'Synergistic cells': bliss['summary']['n_synergistic'],
         'Antagonistic cells': bliss['summary']['n_antagonistic']},
        {'Method': 'ZIP (4PL)',
         'Synergy statistic': 'mean excess vs fitted single agents',
         'Value': round(zipr['summary']['mean_excess'], 3),
         'Synergistic cells': zipr['summary']['n_synergistic'],
         'Antagonistic cells': zipr['summary']['n_antagonistic']},
        {'Method': 'Loewe / Chou-Talalay',
         'Synergy statistic': 'median combination index (CI)',
         'Value': None if not np.isfinite(med_ci) else round(med_ci, 3),
         'Synergistic cells': loewe['summary']['n_synergy'],
         'Antagonistic cells': loewe['summary']['n_antagonism']},
    ]), hide_index=True, use_container_width=True)
    st.caption('Thresholds: |Bliss/ZIP excess| > 0.10; Loewe synergy CI < 0.9, antagonism '
               'CI > 1.1. A checkerboard measures one plate — replicate before claiming synergy.')

    st.subheader('Model expectation versus the plate')
    if ref is None:
        st.info(f'The sealed model has no PRISM AUC for {drug_a} on {line_name}, so a Bliss '
                'expectation cannot be computed. The plate result stands on its own; RSL3 is '
                'a lab-only lane by design.')
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric(f'Model AUC: {drug_a}', f"{ref['auc_a']:.3f}")
        m2.metric(f'Model AUC: {drug_b}', f"{ref['auc_b']:.3f}")
        m3.metric('Model Bliss E_AB', f"{ref['e_ab']:.3f}",
                  help='Additive expectation from the sealed single-agent AUCs.')
        st.plotly_chart(synergy.comparison_chart(ref['e_ab'], observed_top),
                        use_container_width=True, config={'displaylogo': False})
        delta = observed_top - ref['e_ab']
        st.caption(f'Observed minus model-additive effect at the top doses: **{delta:+.3f}**. '
                   'Positive means the plate killed more than the model’s additivity baseline; '
                   'that is the quantity a synergy claim must defend with replicates.')
    with st.expander('How the three scores are defined'):
        st.write('**Bliss** — expected effect E_A + E_B − E_A·E_B, using the measured '
                 'single-agent wells. **ZIP** — the same combination rule but E_A/E_B come '
                 'from 4-parameter logistic fits, which removes plate noise from the '
                 'single-agent curves. **Loewe** — combination index CI = a/A_eq + b/B_eq, '
                 'where A_eq/B_eq are the single-agent doses that would produce the observed '
                 'effect; CI < 1 is synergy. All three are computed from the exact matrix '
                 'above; nothing is imputed.')


def network_tab():
    """Interactive protein rescue network from the expressed-protein shortlist."""
    st.subheader('Interactive protein rescue network')
    st.write('Nodes are proteins: orange = suppressed by the killer, green = the validated '
             'killer↓ ∩ rescue↑ overlap (the in-silico rescue evidence). Killer and rescue '
             'drugs are the large nodes; dotted edges = suppression, solid edges = the '
             'validated reversal overlap. Only the overlap genes stored in the reversal '
             'table are drawn — the full rescue-up signature is reported as a count, not '
             'invented.')
    short_path = os.path.join(DATA, 'protein_array_shortlist.csv')
    if not os.path.exists(short_path):
        st.info('Protein-array shortlist not in this deployment.')
        return
    sl = load('protein_array_shortlist.csv')
    rev_path = os.path.join(DATA, 'offtarget_rescue_reversal.csv')
    rv = load('offtarget_rescue_reversal.csv') if os.path.exists(rev_path) else None
    pm_path = os.path.join(DATA, 'offtarget_panel_map.csv')
    pm = load('offtarget_panel_map.csv') if os.path.exists(pm_path) else None
    cov_path = os.path.join(DATA, 'offtarget_coverage_ferroselect.csv')
    cov = load('offtarget_coverage_ferroselect.csv') if os.path.exists(cov_path) else None

    c1, c2, c3 = st.columns([2, 2, 1])
    killer = c1.selectbox('Killer / combination', network_graph.KILLER_OPTIONS,
                          index=0, key='net_killer')
    rescue = c2.selectbox('Rescue control', network_graph.RESCUE_OPTIONS,
                          index=0, key='net_rescue')
    line_name = c3.selectbox('Cell line (reversal p)', list(LINE_IDS), key='net_line')
    c4, c5 = st.columns([1, 2])
    max_nodes = c4.slider('Maximum protein nodes', 20, 150, 80, step=10, key='net_nodes')
    label_shared = c5.checkbox('Label the shared rescue proteins', value=True, key='net_labels')

    fig, meta = network_graph.rescue_network_figure(
        sl, rv, killer, rescue, LINE_IDS[line_name], max_nodes=max_nodes,
        label_shared=label_shared, panel_map=pm, coverage=cov)
    if not meta['valid']:
        st.error(meta['warning'])
        provenance = pd.DataFrame([
            {'Field': 'Requested rescue agent', 'Value': str(rescue)},
            {'Field': 'L1000 term found by the legacy matcher',
             'Value': str(meta.get('signature_term') or '—')},
            {'Field': 'Suppressed proteins available', 'Value': str(meta['n_suppressed'])},
            {'Field': 'Network status', 'Value': 'WITHHELD (wrong molecule)'},
        ])
        st.dataframe(provenance, hide_index=True, use_container_width=True)
        st.caption('This is a provenance audit, not a rendering failure. A rescue network '
                   'built from the wrong molecule would be worse than no network. Supply a '
                   'genuine NAC signature to enable this arm; DEFEROXAMINE is validated and '
                   'available now.')
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Suppressed proteins', f"{meta['n_suppressed']:,}")
    m2.metric('Rescue-up signature', f"{meta['n_rescued']:,}",
              help='Full size of the rescue agent’s Up signature in expressed genes.')
    m3.metric('Validated overlap', f"{meta['n_overlap']:,}",
              help='Genes stored in offtarget_rescue_reversal.csv (killer-down ∩ rescue-up).')
    m4.metric('Hypergeometric p', 'n/a' if meta['p'] is None else
              ('< 1e-16' if meta['p'] == 0 else f"{meta['p']:.2g}"),
              help='Overlap of killer-suppressed and rescue-up proteins; lower = stronger.')
    st.caption(meta['note'] or f'{meta["n_suppressed"]} suppressed proteins for {killer}.')
    if meta['enrich'] is not None:
        st.caption(f'Enrichment {meta["enrich"]:.2f}× over chance on {line_name}; '
                   f'{meta["shown"]} of {meta["n_suppressed"]} suppressed proteins are drawn '
                   f'(raise the slider to see more). All {meta["n_overlap"]} stored overlap '
                   'proteins are included.')
    st.plotly_chart(fig, use_container_width=True, config={'displaylogo': False})
    st.caption('The shortlist carries the MDA-MB-231 expressed-protein signature; the line '
               'selector changes the hypergeometric reversal statistic, not the killer gene '
               'list. The rescue-up signature is reported from the reversal table; the full '
               'gene list is not stored, so no rescue-only nodes are invented.')

    with st.expander('Signature provenance (audit note)'):
        st.write('**NAC is deliberately withheld.** The legacy coverage run resolved '
                 '`N-ACETYLCYSTEINE` to the L1000 term **Solifenacin-Succinate** because the '
                 'three-letter abbreviation “NAC” is a substring of “SolifeNACin”. The 166 '
                 '“NAC up” genes in the older artifacts therefore belong to a different '
                 'molecule. The dashboard refuses to plot them as NAC; DEFEROXAMINE '
                 '(Deferoxamine-Mesylate) is a valid L1000 term and remains available.')
        if cov is not None and {'panel_drug', 'l1000_term'} <= set(cov.columns):
            st.dataframe(cov[['panel_drug', 'l1000_term', 'l1000_sig_genes']],
                         hide_index=True, use_container_width=True)

    paths_path = os.path.join(DATA, 'network_paths.json')
    if os.path.exists(paths_path):
        with st.expander('Curated mechanism paths to GPX4 / ferroptosis'):
            mfig, _ = network_graph.mechanism_path_figure(
                load_json('network_paths.json'),
                ['erastin', 'sulfasalazine', 'simvastatin', 'lovastatin', 'RSL3'])
            if mfig is not None:
                st.plotly_chart(mfig, use_container_width=True, config={'displaylogo': False})
                st.caption('Shortest routes computed by src/models/network.py; every route '
                           'converges on GPX4 and lipid hydroperoxides.')


@st.cache_data(show_spinner=False)
def _defence_html(stamp):
    del stamp                       # mtime key invalidates the cache when artifacts change
    return defense_report.build_defense_report(DATA)


def defense_tab():
    """One-click, self-contained HTML executive report (print → PDF)."""
    st.subheader('Automated defence report (HTML, print-to-PDF)')
    st.write('Builds the **v1.1 dated addendum** from the committed artifacts: the '
             'leakage controls and 76-line correction, the post-hoc per-drug baseline, '
             'the selectivity null, the 8-member panel, the plateau, Bliss expectations, '
             'the CTRPv2 RSL3 evidence, prediction intervals, validation gates and a '
             'sign-off block. No numbers are typed by hand. v1.0 is preserved at tag '
             '`v1.0-defense`; the sealed prediction is unchanged.')
    try:
        stamp = max(os.path.getmtime(os.path.join(DATA, f)) for f in os.listdir(DATA))
        html_report = _defence_html(stamp)
    except Exception as e:
        st.error('Could not build the report from the current artifacts.')
        st.caption(f'Details: {type(e).__name__}: {e}')
        return
    d1, d2 = st.columns([1, 2])
    with d1:
        st.download_button('Download defence report (HTML)', html_report.encode('utf-8'),
                           'FerroSelect_defence_report.html', 'text/html', key='dl_defence')
    with d2:
        st.caption('Open the downloaded file and use **Print → Save as PDF** (A4, margins '
                   '16 mm). The report is offline-safe: CSS only, no JavaScript, no external '
                   'fonts or images.')
    st.dataframe(pd.DataFrame([
        {'Section': '1', 'Contents': 'Dataset, leakage controls and the 76-line correction'},
        {'Section': '2', 'Contents': 'CatBoost vs per-drug baseline (post-hoc, exploratory)'},
        {'Section': '3', 'Contents': 'Selectivity null and “Predicted selective (unvalidated)”'},
        {'Section': '4', 'Contents': 'The eight-member wet-lab panel, as a mechanism hypothesis'},
        {'Section': '5', 'Contents': 'Top-ranked killers, plateau and sensitive-tail coverage'},
        {'Section': '6', 'Contents': 'Bliss additivity arithmetic — test on the plate'},
        {'Section': '7', 'Contents': 'CTRPv2.1 measured RSL-3 AUCs (external evidence)'},
        {'Section': '8', 'Contents': 'Conformal prediction intervals (±80% / ±95%)'},
        {'Section': '9', 'Contents': 'Validation gates, quarantine and test status'},
        {'Section': '10', 'Contents': 'Signature block for the defence appendix'},
    ]), hide_index=True, use_container_width=True)
    with st.expander('Preview the report'):
        components.html(html_report, height=760, scrolling=True)


# ---------- header ----------
st.title('FerroSelect: which drugs kill TNBC, and by what mechanism')
st.write('Predicted sensitivity for **MDA-MB-231**, compared with the luminal '
         'control **MCF7**, plus the ferroptosis inducer panel from the proposal. '
         'Lower predicted AUC means stronger killing. The v1.1 audit reports the '
         'model as a **negative result**; the wet-lab panel is the hypothesis test.')
st.info('Predicted sensitivity in a cell model, not a clinical recommendation. '
        f"Model version: {info.get('model_version', 'n/a')}")

_cb = ((baseline_audit or {}).get('metrics') or {}).get('catboost') or {}
_bm = ((baseline_audit or {}).get('metrics') or {}).get('per_drug_mean') or {}
_learn = ((baseline_audit or {}).get('subsets') or {}).get('learned_227') or {}
_lcb = (_learn.get('catboost') or {}).get('per_line_spearman')
_lbm = (_learn.get('per_drug_mean') or {}).get('per_line_spearman')
if _cb and _bm:
    learned_text = (f" Within the 227 drugs CatBoost actually learned, it ties the "
                    f"baseline ({_lcb:.3f} vs {_lbm:.3f});" if _lcb and _lbm else '')
    st.warning(
        f"**Post-hoc, exploratory (v1.1):** a one-line per-drug mean baseline beats "
        f"CatBoost on the spent holdout — per-line Spearman **{_bm['per_line_spearman']:.3f} "
        f"vs {_cb['per_line_spearman']:.3f}**, R² **{_bm['r2']:.3f} vs {_cb['r2']:.3f}**. "
        f"NDCG@10 is a tie ({_bm['ndcg10']:.3f} vs {_cb['ndcg10']:.3f})."
        + learned_text +
        " The expression contribution is unproven. Do not cite the champion as a "
        "validated cell-line-specific predictor.")

with st.expander('How to read this dashboard (60 seconds, no ML background needed)'):
    st.write('**AUC** — predicted area under the dose-response curve, 0 to 1. '
             '**Lower means the drug kills more.** Below 0.30 we call it SENSITIVE.')
    st.write('**Selectivity Δ** — MCF7 AUC minus MDA-MB-231 AUC. Positive means the '
             'model predicts the drug hits the TNBC line harder than the control line. '
             'This is a prediction, not a measured selectivity result.')
    st.write('**R² / Spearman / NDCG@10** — how well the model ranked unseen cell lines '
             'during testing. CatBoost met the pre-registered gates (R² > 0.25, per-line '
             'Spearman > 0.4), but a post-hoc per-drug mean baseline beats it on both. '
             'Read the gates as “not useless”, not as “validated”.')
    st.write('**LOW confidence** — fewer than 25 training rows for that drug. A hypothesis '
             'to test, not a result.')
    st.write('**Bliss pairs** — expected combined effect if two drugs simply add up. '
             'A synergy claim needs the checkerboard experiment, never this table.')
    st.write('**Pending annotation** — pathway identity is not yet curated by the bio '
             'informatics side; the pair itself is still valid.')

# ---------- metric cards with honest deltas ----------
def delta_str(value, base_row, key):
    """Delta against a real baseline row; never invent one."""
    if base_row is None or pd.isna(base_row.get(key, np.nan)):
        return None
    d = value - base_row[key]
    if abs(d) < 0.005:
        return f'tie vs {base_row.name}'
    return f'{d:+.2f} vs {base_row.name}'


def gate_str(value, gate):
    """Distance to a pre-registered acceptance gate."""
    return f'{value - gate:+.2f} vs {gate:.2f} target'


ACCEPTANCE_GATES = {'r2': 0.25, 'per_line_spearman': 0.40}


dummy = lb.loc['DummyMean'] if 'DummyMean' in lb.index else None

with st.container(border=True):
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Champion (v1.0)', champion,
              help='Pre-registered model-selection winner. Champion is a convention, '
                   'not a finding: the gap to XGBoost is within 7-line resolution.')
    k2.metric('CatBoost per-line Spearman', f"{(_cb or best)['per_line_spearman']:.3f}",
              help='Pre-registered holdout metric. Higher is better.')
    k3.metric('Per-drug mean baseline', f"{(_bm or {}).get('per_line_spearman', float('nan')):.3f}",
              help='Post-hoc, exploratory. Each drug’s mean training AUC; global-mean '
                   'fallback for unseen drugs (0 rows needed it here).')
    k4.metric('Δ CatBoost − baseline',
              f"{(_cb or best)['per_line_spearman'] - (_bm or {'per_line_spearman': float('nan')})['per_line_spearman']:+.3f}",
              help='Paired bootstrap over the seven holdout lines; negative means the '
                   'baseline ranks better. NDCG@10 is a tie.')
    st.caption('Pre-registered gates on CatBoost: R² > 0.25 '
               f"({'PASS' if best['r2'] > ACCEPTANCE_GATES['r2'] else 'FAIL'}, "
               f"{best['r2']:.3f}) and per-line Spearman > "
               f"{ACCEPTANCE_GATES['per_line_spearman']:.2f} "
               f"({'PASS' if best['per_line_spearman'] > ACCEPTANCE_GATES['per_line_spearman'] else 'FAIL'}, "
               f"{best['per_line_spearman']:.3f}). **But the post-hoc per-drug mean "
               f"baseline wins both** (R² {(_bm or {}).get('r2', float('nan')):.3f}; "
               f"per-line {(_bm or {}).get('per_line_spearman', float('nan')):.3f}); "
               'the model is reported as a negative result.')

# ---------- sidebar ----------
with st.sidebar:
    st.header('About this run')
    st.markdown(
        f"**Training pairs:** {info.get('training_pairs', 30994):,}  \n"
        "30 lines × 1,482 drugs, deduplicated  \n\n"
        f"**Holdout lines:** {info.get('holdout_lines', 7)}  \n"
        "Includes MDA-MB-231 and MCF7, which were never tuned on.")
    st.caption('Data: DepMap 26Q3, GDSC2, PRISM-REPURPOSING 27Oct23. '
               'See model_info.json and audit_summary.json for versions.')

# ---------- tabs (7) ----------
tab_rank, tab_sel, tab_syn, tab_net, tab_model, tab_report, tab_live = st.tabs(
    ['Rankings & candidates', 'Selectivity & combinations', 'Synergy checkerboard',
     'Protein network', 'Model & audit', 'Defense report', 'Live predictor'])

# ---------------- 1. rankings ----------------
with tab_rank:
    st.subheader('Ranked drug candidates')
    st.caption(selectivity_caption())
    view = st.radio('Dataset', ['Full ranking', 'Top-20 killers', 'Wet-lab shortlist',
                                'FerroSelect inducer panel'],
                    horizontal=True, label_visibility='collapsed')

    if view == 'Wet-lab shortlist':
        sl = shortlist.copy()
        roles = []
        for i, name in enumerate(sl['drug_name']):
            if str(name).lower() in POSITIVE_CONTROLS:
                roles.append('Positive control')
            elif i == len(sl) - 1:
                roles.append('Predicted resistant')
            else:
                roles.append('AI candidate')
        sl.insert(0, 'role', roles)
        st.caption('Roles: AI candidate = model-nominated; Positive control = known killer '
                   'that validates the assay; Predicted resistant = negative control that should survive.')
        dl_c, _ = st.columns([1, 3])
        with dl_c:
            st.download_button('Download CSV', to_csv(sl), 'wetlab_shortlist.csv',
                               'text/csv', key='dl_sl')
        rank_table(sl[['role', 'drug_name', 'pred_AUC_MDA231', 'auc_ci_mda231', 'pred_AUC_MCF7',
                       'selectivity_delta', 'sensitive_call', 'selective_call']])
        st.subheader('Starting dose brackets (proposal §6.4 — narrow them in pilot)')
        st.caption('Vehicle capped at 0.1% DMSO; every plate carries a vehicle-matched control.')
        st.dataframe(pd.DataFrame([
            ['Sulfasalazine', 'Single - Type I inducer (workhorse)', '50–1000 µM, 8 points', 'Cheap generic'],
            ['Erastin', 'Single - Type I reference', '0.1–20 µM (published IC50 ~2.2)', 'Research-grade import'],
            ['RSL3', 'Single - Type II killer (GPX4)', '0.01–5 µM, MDA-MB-231 only', 'Lab-only - no PRISM data; see panel table'],
            ['Erastin + Simvastatin', 'Combo', 'Erastin dose-response × statin ~IC20', 'Bliss-predicted pair'],
            ['Sulfasalazine + Simvastatin', 'Combo (sealed nomination)', 'Sulfasalazine dose-response × statin ~IC20', 'Top sealed prediction'],
            ['RSL3 + Simvastatin', 'Combo', 'RSL3 low-dose × statin ~IC20', 'Lab-only - no PRISM data'],
            ['Simvastatin', 'Sensitiser inside the combos only', 'Sub-lethal ~IC20, set in pilot', 'Cheap generic; never a single agent'],
            ['Ferrostatin-1', 'Rescue control', '1–2 µM alongside', 'Must rescue if ferroptosis'],
            ['Deferoxamine (DFO)', 'Rescue control', '50–100 µM alongside', 'Death needed iron? Reverses 4/5 killers'],
            ['N-acetylcysteine (NAC)', 'Rescue control', '1–5 mM alongside', 'Rescues Type I, not RSL3'],
            ['Z-VAD-FMK', 'Specificity control', '20–50 µM alongside', 'Must NOT rescue'],
        ], columns=['Agent', 'Role in the assay', 'Starting range', 'Note']),
            hide_index=True, use_container_width=True)

    elif view == 'FerroSelect inducer panel':
        st.caption('Proposal-faithful predictions (Lasso/Ridge/RF, nested cell-line CV) for '
                   'the inducer panel on the three proposal lines. LOW confidence = fewer '
                   'than 25 training rows; a hypothesis for the lab, not evidence.')

        st.subheader('The wet-lab panel (8 members)')
        panel = pd.DataFrame([
            ['ERASTIN', 'Single - killer (Type I, system xc-)',
             'PRISM AUC + L1000 signature'],
            ['SULFASALAZINE', 'Single - killer (Type I)',
             'PRISM AUC + L1000 signature'],
            ['RSL3', 'Single - killer (Type II, GPX4)',
             'GSE247883 single-paper signature; NOT in PRISM/L1000 (lab-only)'],
            ['N-ACETYLCYSTEINE (NAC)', 'Single - rescue control',
             'L1000 signature (reversal control)'],
            ['DEFEROXAMINE (DFO)', 'Single - rescue control',
             'L1000 signature (reversal control)'],
            ['ERASTIN + SIMVASTATIN', 'Combo',
             'PRISM E_AB + L1000 signatures'],
            ['SULFASALAZINE + SIMVASTATIN', 'Combo - sealed nomination',
             'PRISM E_AB + L1000 signatures'],
            ['RSL3 + SIMVASTATIN', 'Combo',
             'Lab-only (the RSL3 side has no PRISM/L1000 data)'],
        ], columns=['Panel member', 'Role', 'Data status'])
        st.dataframe(panel, hide_index=True, use_container_width=True)
        st.caption('The 8 members agreed with the lab. The predictions table below covers '
                   'only the members PRISM has AUC data for; the RSL3 rows are tracked '
                   'through the GSE247883 single-paper signature instead.')

        if os.path.exists(os.path.join(DATA, 'ferroselect_panel_predictions.csv')):
            preds = load('ferroselect_panel_predictions.csv')
            preds = preds[preds['drug'] != 'LOVASTATIN']  # panel uses SIMVASTATIN only
            st.subheader('Panel ranking 1-8 on MDA-MB-231 (the eight wet-lab members)')
            prank = panel_ranking_table(preds)
            st.dataframe(prank, hide_index=True, use_container_width=True,
                         column_config={'Predicted AUC (MDA-MB-231)':
                                        st.column_config.NumberColumn(format='%.3f')})
            st.plotly_chart(panel_ranking_chart(prank), use_container_width=True,
                            config={'displaylogo': False})
            st.caption('Rank 1 = strongest expected killing. Where the model has a number the '
                       'order follows the sealed predicted AUC (lower = stronger). RSL3 and '
                       'its combo carry no PRISM AUC (lab-only lanes, shaded in the note) and '
                       'the two rescue controls sit last because they are not killers — they '
                       'must bring the death back, not add to it.')

            # CTRPv2.1 external ground truth for RSL3 (measured AUCs, v2 track)
            if os.path.exists(os.path.join(DATA, 'ctpv2_ferroptosis_auc.csv')):
                st.subheader('CTRPv2 external ground truth for RSL3 (measured, not modelled)')
                ctp = load('ctpv2_ferroptosis_auc.csv')
                gold_map = {'ACH-000768': 'MDA-MB-231', 'ACH-000849': 'MDA-MB-468',
                            'ACH-000019': 'MCF7'}
                rsl_g = ctp[(ctp['target'] == 'RSL3') & ctp.ModelID.isin(gold_map)].copy()
                rsl_g['line'] = rsl_g.ModelID.map(gold_map)
                rsl_g = rsl_g.sort_values('auc_norm_0_1')
                gx = rsl_g[['line', 'auc_raw', 'auc_norm_0_1', 'sens_percentile']]
                gx.columns = ['Cell line', 'RSL-3 raw AUC (CTRPv2)',
                              'AUC on 0-1 scale', 'Percentile most sensitive (of 808 lines)']
                st.dataframe(gx, hide_index=True, use_container_width=True,
                             column_config={'RSL-3 raw AUC (CTRPv2)':
                                            st.column_config.NumberColumn(format='%.2f'),
                                            'AUC on 0-1 scale':
                                            st.column_config.NumberColumn(format='%.3f')})
                note2 = ('RSL-3 was profiled in CTRPv2.1 (Nat Chem Biol 2016): integrated '
                         'area under the concentration-response curve, LOWER = more cell '
                         'death. MDA-MB-231 sits in the most-sensitive 30% of the 808-line '
                         'cohort while MCF7/MB-468 are mid-pack - the mesenchymal/GPX4 '
                         'dependency pattern Viswanathan 2017 published. The v2 model '
                         '(Lasso/Ridge/RF on the same pca features v1 used, GroupKFold '
                         'OOF on 804 lines) tracks this cohort at Spearman ~0.46 — '
                         'expression-only signal, far below the 0.83 published lab-vs-CTRP '
                         'ceiling. Sealed v1 files are untouched; these AUCs are real '
                         'measurements, not predictions.')
                if os.path.exists(os.path.join(DATA, 'ctpv2_meta.json')):
                    md3 = load_json('ctpv2_meta.json').get('rsl3_v2', {})
                    sp = md3.get('models', {}).get(md3.get('winner', ''), {}).get('spearman')
                    if sp is not None:
                        note2 = note2.replace('Spearman ~0.46',
                                              f'Spearman {sp:.3f} ({md3["winner"]})')
                st.caption(note2)
            st.subheader('Combination Bliss ranking - the three Simvastatin pairs')
            st.plotly_chart(combo_bliss_chart(preds), use_container_width=True,
                            config={'displaylogo': False})
            st.caption('Expected combined effect per line from the sealed FerroSelect AUCs. '
                       'RSL3 + SIMVASTATIN cannot be Bliss-scored (no PRISM AUC for RSL3); '
                       'it stays a wet-lab-only question. Every value here inherits the LOW '
                       'confidence of 9-22 training rows per agent.')
            fpred = preds.copy()
            if 'pred_auc' in fpred.columns:
                fpred['pred_auc'] = fpred['pred_auc'].astype(float).round(3)
                line_id_map = {'MDA-MB-231': 'ACH-000768', 'MDA-MB-468': 'ACH-000849',
                               'MCF7': 'ACH-000019'}
                fpred.insert(
                    list(fpred.columns).index('pred_auc') + 1, 'pred_auc ± 80% CI',
                    [ci_label(a, line_id_map.get(str(ln)), did)
                     for a, ln, did in zip(fpred['pred_auc'], fpred.get('line', ''),
                                           fpred.get('drug_id', pd.Series([None] * len(fpred))))])
            if 'confidence' in fpred.columns:
                fpred['confidence'] = fpred['confidence'].astype(str).map(
                    lambda v: 'LOW (<25 rows)' if v.strip().upper().startswith('LOW') else v)
            dl_c, _ = st.columns([1, 3])
            with dl_c:
                st.download_button('Download CSV', to_csv(fpred),
                                   'ferroselect_panel_predictions.csv', 'text/csv', key='dl_fs')
            st.dataframe(fpred, hide_index=True, use_container_width=True)
            st.caption('RSL3 has no row in the predictions table by design - PRISM has no '
                       'RSL3 data; the panel table above and the signature tables below '
                       'track it instead. LOVASTATIN rows are also hidden here on purpose: '
                       'the sealed panel uses SIMVASTATIN only (LOVASTATIN stays in the '
                       'downloaded CSV as the PRISM-screened statin alternative).')
        else:
            st.info('FerroSelect predictions are not in this deployment yet.')

        rescue_path = os.path.join(DATA, 'offtarget_rescue_reversal.csv')
        shortlist_path = os.path.join(DATA, 'protein_array_shortlist.csv')
        if os.path.exists(rescue_path):
            st.subheader('In-silico rescue reversal (protein level)')
            st.caption('Do the rescue controls re-express the proteins each killer '
                       'suppresses? Hypergeometric test on the expressed-protein overlap. '
                       'NAC rows are omitted: the archived NAC signature is a Solifenacin '
                       'artefact from a legacy substring match (provenance audit); '
                       'Deferoxamine is the validated rescue arm.')
            res = load('offtarget_rescue_reversal.csv')
            res = res[(res['killer'] != 'LOVASTATIN')
                      & (res['rescue'].map(lambda v: str(v).upper())
                         != 'N-ACETYLCYSTEINE')]      # NAC signature is mislabelled
            res['line'] = res['line'].map(lambda v: LINE_NAMES.get(v, v))
            res['signature_source'] = res['signature_source'].fillna('LINCS L1000')
            res = res.rename(columns={'line': 'Line', 'killer': 'Killer',
                                      'rescue': 'Rescue drug',
                                      'killer_down_genes': 'Proteins suppressed',
                                      'rescue_up_genes': 'Rescue up-regulates',
                                      'reversal_hits': 'Re-expressed',
                                      'reversal_p': 'p (hypergeom)',
                                      'reversal_enrich': 'Enrichment x',
                                      'genes': 'Shared genes',
                                      'signature_source': 'Signature source'})
            res['p (hypergeom)'] = res['p (hypergeom)'].map(
                lambda v: '< 1e-16' if not v or v == 0 else f'{v:.2g}')
            res['Enrichment x'] = res['Enrichment x'].astype(float).round(2)
            dl_c, _ = st.columns([1, 3])
            with dl_c:
                st.download_button('Download CSV', to_csv(res),
                                   'offtarget_rescue_reversal.csv', 'text/csv', key='dl_res')
            st.dataframe(res, hide_index=True, use_container_width=True)
        else:
            st.info('Rescue-reversal table not in this deployment '
                    '(run src/models/offtarget_panel.py + rsl3_signature.py).')
        if os.path.exists(shortlist_path):
            st.subheader('Protein-array shortlist')
            st.caption('Candidates to print on the wet-lab protein array: rescue-competent '
                       'first (they re-express under rescue controls), then '
                       'membrane/secreted proteins that are easy to assay.')
            sl2 = load('protein_array_shortlist.csv').rename(columns={
                'drug': 'Drug', 'gene': 'Gene', 'in_rescue_reversal': 'Rescue-competent',
                'array_friendly': 'Array-friendly', 'in_statin_core': 'In statin core set'})
            sl2['Array-friendly'] = sl2['Array-friendly'].fillna('intracellular')
            core = sl2[sl2['Rescue-competent'].astype(str).isin(['1', '1.0', 'True'])]
            dl_c, _ = st.columns([1, 3])
            with dl_c:
                st.download_button('Download CSV', to_csv(sl2), 'protein_array_shortlist.csv',
                                   'text/csv', key='dl_short')
            st.dataframe(core, hide_index=True, use_container_width=True, column_config={
                'Rescue-competent': st.column_config.CheckboxColumn(
                    'Rescue-competent', disabled=True,
                    help='Also re-expressed under a rescue control in silico.'),
                'In statin core set': st.column_config.CheckboxColumn(
                    'In statin core set', disabled=True,
                    help='Suppressed by the statin in its L1000 signature core set.')})
            st.caption(f'Showing the {len(core)} rescue-competent candidates; the CSV '
                       f'contains all {len(sl2)} candidates with the array-friendly and '
                       f'statin-core flags.')

            st.subheader('The rescue layer, in pictures')
            line_sel = st.selectbox('Line', ['MDA-MB-231', 'MDA-MB-468', 'MCF7'],
                                    key='sel_rev_line')
            code_sel = {'MDA-MB-231': 'ACH-000768', 'MDA-MB-468': 'ACH-000849',
                        'MCF7': 'ACH-000019'}[line_sel]

            st.write('**Rescue power at a glance** — how strongly each rescue control '
                     're-expresses the proteins each killer suppresses on the chosen '
                     'line. Bubble size = re-expressed proteins; colour = rescue drug.')
            rev = load('offtarget_rescue_reversal.csv')
            rev = rev[(rev['killer'] != 'LOVASTATIN') & (rev['line'] == code_sel)]
            st.plotly_chart(rescue_bubble(rev), use_container_width=True,
                            config={'displaylogo': False})
            st.caption('RSL3 comes from the GSE247883 single-paper signature, so it is '
                       'plotted on MDA-MB-231 only. DFO does not reverse RSL3 '
                       '(p = 0.41, not significant) — exactly the mechanism split the '
                       'wet lab should see. Exact p values: table above.')

            st.write('**Where the array candidates sit** — the rescue-competent proteins, '
                     'split by location. Membrane and secreted proteins go straight on '
                     'the array; intracellular ones need cell lysis.')
            st.plotly_chart(array_priority_bar(core), use_container_width=True,
                            config={'displaylogo': False})

    else:
        df = rank_full if view == 'Full ranking' else top20
        k, search = st.columns([1, 2])
        n_k = k.slider('Show top K drugs', 5, 50, 20)
        query = search.text_input('Find a drug', placeholder='e.g. paclitaxel')
        hits = df[df['drug_name'].str.contains(query, case=False, na=False, regex=False)] if query else df.head(n_k)
        if query:
            st.caption(f'{len(hits)} match(es) for "{query}"')
        dl_c, _ = st.columns([1, 3])
        with dl_c:
            st.download_button('Download CSV', to_csv(df),
                               'mda_mb_231_full_ranking.csv' if view == 'Full ranking'
                               else 'mda_mb_231_top20_killers.csv', 'text/csv', key='dl_top')
        rank_table(hits[RANK_COLS])
        st.subheader(f'Predicted AUC of the top {min(n_k, len(hits))} (lower is stronger)')
        st.plotly_chart(bar_chart(hits[RANK_COLS]), use_container_width=True,
                        config={'displaylogo': False})
        st.caption('Whiskers are 80% conformal prediction intervals calibrated on the seven '
                   'blind holdout lines (MDA-MB-231 half-width ±0.093; global ±0.104). '
                   'They describe model uncertainty, not assay noise — the plate decides.')

# ---------------- 2. selectivity + combos ----------------
with tab_sel:
    st.subheader('Selectivity: MDA-MB-231 vs MCF7')
    st.warning(selectivity_caption())
    st.write(f'Below the dashed diagonal means the drug is predicted to kill the TNBC line '
             f'more than the control. The dotted lines mark the sensitivity cut-off '
             f'(AUC {SENSITIVE_AUC}). Hover for values; drag to zoom. These are model '
             'predictions; measured selectivity requires the SI from the plate.')
    n_top = st.slider('Label the top N drugs', 0, 10, 5)
    st.plotly_chart(selectivity_chart(rank_full, n_top), use_container_width=True,
                    config={'displaylogo': False})

    st.subheader('Two-drug combinations (Bliss independence)')
    st.write('Ranked by selectivity gap: expected efficacy on MDA-MB-231 minus expected '
             'efficacy on MCF7. This assumes additivity, so a synergy verdict needs the '
             '4×4 checkerboard validation.')
    cb = combos.copy()
    for c in ['E_AB_MDA231', 'E_AB_MCF7', 'selectivity_gap']:
        cb[c] = cb[c].astype(float).round(3)
    for c in ['pathwayA', 'pathwayB', 'pathway_check']:
        cb[c] = cb[c].astype(str).replace({'?': PENDING, 'PENDING-Bioinfo': PENDING})
    cb = cb.rename(columns={'pathwayA': 'Pathway A', 'pathwayB': 'Pathway B',
                            'pathway_check': 'Pathway check', 'E_AB_MDA231': 'E_AB MDA-MB-231',
                            'E_AB_MCF7': 'E_AB MCF7'})
    cb_styled = badge_styler(cb, ['Pathway A', 'Pathway B', 'Pathway check'], PENDING_PALETTE)
    st.dataframe(cb_styled, hide_index=True, use_container_width=True,
                 column_config={'Pathway A': st.column_config.TextColumn('Pathway A'),
                                'Pathway B': st.column_config.TextColumn('Pathway B'),
                                'Pathway check': st.column_config.TextColumn('Pathway check',
                                                                             width='medium')})

# ---------------- 3. synergy checkerboard ----------------
with tab_syn:
    synergy_tab()

# ---------------- 4. protein network ----------------
with tab_net:
    network_tab()

# ---------------- 5. model & audit ----------------
with tab_model:
    st.subheader('Holdout benchmark: CatBoost vs a one-line baseline')
    st.caption('**Post-hoc, exploratory.** The holdout was already seen when this '
               'baseline was defined, so it is not a pre-registered result. Both models '
               'are scored on the identical 7,336 holdout rows (7 cell lines).')
    bshow = baseline_table.copy()
    for c in ['mae', 'rmse', 'r2', 'pooled_spearman', 'per_line_spearman',
              'ndcg10', 'recall10']:
        if c in bshow.columns:
            bshow[c] = bshow[c].astype(float).round(3)
    bshow = bshow.rename(columns={
        'model': 'Model', 'analysis_status': 'Analysis status', 'n': 'n rows',
        'mae': 'MAE', 'rmse': 'RMSE', 'r2': 'R²',
        'pooled_spearman': 'Pooled Spearman',
        'per_line_spearman': 'Per-line Spearman', 'ndcg10': 'NDCG@10',
        'recall10': 'Recall@10'})
    st.dataframe(bshow[['Model', 'Analysis status', 'n rows', 'MAE', 'RMSE', 'R²',
                        'Pooled Spearman', 'Per-line Spearman', 'NDCG@10',
                        'Recall@10']],
                 hide_index=True, use_container_width=True)

    if baseline_audit:
        paired = baseline_audit.get('paired_bootstrap_catboost_minus_baseline') or {}
        p1, p2, p3 = st.columns(3)
        for col, key, label in [(p1, 'per_line_spearman', 'Per-line Spearman'),
                                (p2, 'r2', 'R²'),
                                (p3, 'ndcg10', 'NDCG@10')]:
            v = paired.get(key) or {}
            if v:
                col.metric(f'Δ {label}', f"{v['mean']:+.3f}",
                           delta=f"95% CI {v['lo95']:+.3f} to {v['hi95']:+.3f}",
                           delta_color='off')
        st.caption('Δ = CatBoost − per-drug mean. Negative favors the baseline. The '
                   'paired bootstrap resamples the seven holdout **lines**, not rows, '
                   'because rows within a line are not independent. NDCG@10 is saturated '
                   'near 0.94 for both models; its interval spans zero and is not evidence '
                   'for either model.')

        with st.expander('Subsets, calibration and sensitive-tail coverage'):
            rows = []
            for name, models in (baseline_audit.get('subsets') or {}).items():
                cb = models['catboost']
                bm = models['per_drug_mean']
                rows.append({'Subset': name.replace('_', ' '), 'n': cb['n'],
                             'CatBoost per-line': round(cb['per_line_spearman'], 3),
                             'Baseline per-line': round(bm['per_line_spearman'], 3),
                             'CatBoost R²': round(cb['r2'], 3),
                             'Baseline R²': round(bm['r2'], 3),
                             'CatBoost NDCG': round(cb['ndcg10'], 3),
                             'Baseline NDCG': round(bm['ndcg10'], 3)})
            st.write('**Subsets** — the learned 227 is a tie, not a lift.')
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            cal = baseline_audit.get('calibration') or {}
            if cal:
                crows = []
                for name, c in cal.items():
                    crows.append({'Subset': name.replace('_', ' '), 'n': c['n'],
                                  'Measured mean AUC': round(c['measured_mean_auc'], 3),
                                  'CatBoost mean': round(c['catboost_mean_auc'], 3),
                                  'Baseline mean': round(c['per_drug_mean_auc'], 3)})
                st.write('**Calibration** — CatBoost overpredicts AUC on the drugs it learned.')
                st.dataframe(pd.DataFrame(crows), hide_index=True,
                             use_container_width=True)
            cov = baseline_audit.get('coverage') or {}
            if cov:
                st.write('**Sensitive-tail coverage** — the plateau does not hide the actives.')
                st.write(f"{cov['measured_sensitive_in_learned']} of "
                         f"{cov['measured_sensitive_rows']} measured AUC < 0.3 rows are "
                         f"inside the {cov['learned_drugs']} learned drugs. Every line’s "
                         'measured top-10 is 10/10 inside that set. This is coverage, '
                         'not lift: the baseline covers them trivially.')

    st.subheader('Pre-registered model-selection leaderboard (v1.0)')
    lb_show = leaderboard.copy()
    for c in ['mae', 'rmse', 'r2', 'spearman', 'per_line_spearman', 'ndcg10', 'recall10']:
        lb_show[c] = lb_show[c].astype(float).round(3)
    st.dataframe(lb_show, hide_index=True, use_container_width=True)
    st.caption('The v1.0 leaderboard remains unchanged. The per-drug mean row lives in '
               '`baseline_audit.csv` as a separate post-hoc artifact, so the v1.0 '
               'selection record is not silently rewritten.')

    st.subheader('Prediction uncertainty (80% and 95% intervals)')
    if not intervals:
        st.info('`prediction_intervals.json` is not in this deployment. '
                'Run `python src/models/prediction_intervals.py`.')
    else:
        glob = intervals.get('global') or {}
        u1, u2, u3 = st.columns(3)
        u1.metric('80% half-width (global)', f"±{glob.get('abs_q80', float('nan')):.3f}",
                  help='Conformal-style empirical quantile of holdout residuals.')
        u2.metric('95% half-width (global)', f"±{glob.get('abs_q95', float('nan')):.3f}",
                  help='Use for decision limits; intervals are marginal, not conditional.')
        u3.metric('Calibration rows', f"{int(glob.get('n', 0)):,}",
                  help='Out-of-sample rows on the seven pre-registered holdout cell lines.')
        ci_rows = []
        for mid, s in (intervals.get('lines') or {}).items():
            ci_rows.append({'Cell line': LINE_NAMES.get(mid, mid), 'ModelID': mid,
                            'n': s.get('n'), 'RMSE': round(float(s.get('rmse', np.nan)), 4),
                            '±80% CI': round(float(s.get('abs_q80', np.nan)), 4),
                            '±95% CI': round(float(s.get('abs_q95', np.nan)), 4)})
        st.dataframe(pd.DataFrame(ci_rows).sort_values('Cell line'), hide_index=True,
                     use_container_width=True)
        st.caption('Every AUC shown in the rankings and panel tabs carries the 80% half-width '
                   'for its cell line; per-drug intervals are used when a drug has at least '
                   f'{intervals.get("min_drug_n", 40)} holdout rows, otherwise the line/global '
                   'fallback applies. Intervals quantify model error, not experimental noise. '
                   'The MDA-MB-231 prediction distribution is compressed above ~0.906 for the '
                   'resistant bulk (84.7% of drugs are tied at the same plateau): ranks below '
                   '~226 are not a fine-grained order, and the sensitive tail is where the '
                   'model carries signal.')

    st.subheader('Audit: ceiling check and circularity (Aim 0)')
    a1, a2 = st.columns(2)
    with a1:
        st.write('**Ceiling check** — if a model beat published lab-vs-CTRP agreement '
                 '(~0.83) it would mean leakage, not success.')
        if os.path.exists(os.path.join(DATA, 'ferroselect_results.json')):
            fres = load_json('ferroselect_results.json')
            st.success(fres['ceiling_flag'])
        else:
            st.info('Not available in this deployment.')
    with a2:
        st.write('**Circularity (Aim 0)** — same models and splits, honest measured-AUC '
                 'labels versus ssGSEA labels computed from the input expression itself.')
        if os.path.exists(os.path.join(DATA, 'ferroselect_phase2.json')):
            ph2 = load_json('ferroselect_phase2.json')
            rows = [{'model': m,
                     'measured-AUC Spearman': round(ph2['real_labels'][m]['spearman'], 3),
                     'circular-label Spearman': round(ph2['aim0_labels'][m]['spearman'], 3),
                     'inflation gap': round(ph2['inflation_gap'][m], 3)}
                    for m in ['Lasso', 'Ridge', 'RandomForest']]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info('Run `python src/models/ferroselect_phase2.py` to generate.')

    if os.path.exists(os.path.join(DATA, 'ferroselect_results.json')):
        fres = load_json('ferroselect_results.json')
        fbest = fres['per_model_oof'][fres['frozen_model']]
        st.subheader('FerroSelect frozen model (proposal-mandated, sealed pre-lab)')
        f1, f2, f3 = st.columns(3)
        f1.metric('Frozen model', fres['frozen_model'], help='Proposal-mandated algorithm.')
        f2.metric('Pooled OOF Spearman', f"{fbest['spearman']:.3f}",
                  help='Honest performance on unseen lines. Wide data, small panel.')
        f3.metric('Ceiling check', 'PASS' if 'OK' in fres['ceiling_flag'] else 'REVIEW',
                  help='Above published lab-vs-CTRP agreement (~0.83) would mean leakage.')
        st.caption(f"What to do with this: the model was frozen and sealed before the first "
                   f"plate; {fbest['spearman']:.2f} is honestly low because the panel agents "
                   f"have 9-22 training rows each - so every prediction is a pre-registered "
                   f"hypothesis, the ceiling check rules out leakage, and the lab decides. "
                   f"{fres['ceiling_flag']}")

    st.subheader('Training-data support per agent')
    audit_path = os.path.join(DATA, 'ferroptosis_sample_audit.csv')
    if os.path.exists(audit_path):
        st.dataframe(load('ferroptosis_sample_audit.csv'), hide_index=True, use_container_width=True)

    st.subheader('SHAP explanations')
    load_path = os.path.join(DATA, 'pca_loadings_top.csv')
    have_loadings = os.path.exists(load_path)
    if have_loadings:
        st.caption('`pca_N` labels in the figures map to real genes via the table below: '
                   'each principal component is annotated with its top-3 loading genes '
                   '(loadings verified per component against the shipped PCA scores). '
                   'Components with zero explained variance are the residual direction '
                   'of the sample space — unattributable and flagged in the table.')
        st.caption('Feature names in these figures are the raw model inputs. '
                   '`drug_DPC-xxxxxx` is the internal DepMap/PRISM ID — see the mapping '
                   'below. `pca_N` is an expression component, not a gene.')
    col_a, col_b = st.columns(2)
    for col, fname, title, alt_text in [
        (col_a, 'shap_global_top20.png', 'Global feature impact (Top-20)',
         'SHAP summary of the 20 features with the largest effect on predicted AUC'),
        (col_b, 'shap_local_mda231_top1.png', 'Local explanation: MDA-MB-231 × top killer',
         'SHAP breakdown for MDA-MB-231 and the top-ranked drug')]:
        with col:
            st.write(title)
            path = os.path.join(DATA, fname)
            if os.path.exists(path):
                st.image(path, caption=alt_text, use_column_width=True)
            else:
                st.caption(f'`{fname}` not found in the outputs folder.')

    with st.expander('Feature ID → readable name (drug IDs and PCA genes resolved)'):
        name_of = dict(zip(rank_full['drug_id'].astype(str), rank_full['drug_name']))
        st.write('Any `drug_*` axis label in the SHAP figures can be translated with this map.')
        st.dataframe(pd.DataFrame([{'model feature': f'drug_{k}', 'drug name': v,
                                    'internal ID': k} for k, v in sorted(name_of.items())]),
                    hide_index=True, use_container_width=True)
        if have_loadings:
            ldf = pd.read_csv(load_path)
            st.write('Each expression component, annotated with its top-3 loading genes:')
            st.dataframe(ldf[['pca_feature', 'label', 'explained_variance_ratio',
                              'gene_1', 'gene_2', 'gene_3']],
                        hide_index=True, use_container_width=True)
            st.caption('Read `pca_8 (GENE1, GENE2, GENE3)` as: the third-highest-variance '
                       'genes driving that component. Loadings come from refitting the '
                       'identical pipeline and verifying it against the shipped scores.')
        else:
            st.caption('PC → gene annotations are not generated yet (no loadings file).')

    st.subheader('Mechanism graph (computed from proposal §2.2/Fig.2)')
    st.write('Read left to right: drugs hit targets, targets feed processes, and both '
             'routes converge on GPX4 and lipid hydroperoxides.')
    if os.path.exists(os.path.join(DATA, 'ferroselect_network.png')):
        st.image(os.path.join(DATA, 'ferroselect_network.png'), use_column_width=True)

    st.subheader('How the rescue proves ferroptosis (proposal Fig.2)')
    st.write('Ferrostatin-1 rescues → the death was ferroptosis. Deferoxamine rescues → it '
             'needed iron. NAC rescues Type I inducers but not RSL3. Z-VAD-FMK must NOT '
             'rescue — if it does, the death was apoptosis.')

# ---------------- 6. defence report ----------------
with tab_report:
    defense_tab()

# ---------------- 7. live ----------------
with tab_live:
    live_tab()

st.divider()
st.caption('Reproducibility: DepMap 26Q3 + GDSC2/PRISM-REPURPOSING 27Oct23. '
           'Splits are cell-line holdouts (MDA-MB-231 and MCF7 never tuned on).')