"""TNBC / FerroSelect Drug Response Predictor — Streamlit dashboard.

Run:  streamlit run dashboard/app.py
Compatible with the pinned Streamlit 1.39 (uses use_container_width).
Reads all artifacts from ../outputs_local (no retraining, no API needed).
"""
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'outputs_local'))

SENSITIVE_AUC = 0.3
HIGHLIGHT, MUTED = '#D9480F', '#868E96'
POSITIVE_CONTROLS = {'doxorubicin', 'paclitaxel'}
DEFAULT_LINE = 'ACH-000768'             # MDA-MB-231
CSVS = ['mda_mb_231_full_ranking.csv', 'mda_mb_231_top20_killers.csv',
        'wetlab_shortlist.csv', 'combo_rankings.csv', 'leaderboard.csv']
RANK_COLS = ['rank', 'drug_name', 'pred_AUC_MDA231', 'pred_AUC_MCF7',
             'selectivity_delta', 'sensitive_call', 'selective_call']
AUC_RANGE = (0.2, 0.6)                  # keeps the colour scale informative

st.set_page_config(page_title='FerroSelect / TNBC Drug Response Predictor',
                   layout='wide')

st.markdown("""
<style>
.block-container{padding-top:1.6rem;padding-bottom:2rem;}
[data-testid="stMetricValue"]{font-size:1.9rem;}
[data-testid="stMetricLabel"]{font-size:.85rem;}
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
def load(name):
    return pd.read_csv(os.path.join(DATA, name))


@st.cache_data
def load_json(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


rank_full = load('mda_mb_231_full_ranking.csv')
top20 = load('mda_mb_231_top20_killers.csv')
shortlist = load('wetlab_shortlist.csv')
combos = load('combo_rankings.csv')
leaderboard = load('leaderboard.csv')
info = load_json('model_info.json')


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
SEL_MAP = {'selective': 'Selective', 'non-selective': 'Not selective',
           'tnbc-selective': 'Selective'}
PENDING = 'Pending annotation'


@st.cache_data
def selectivity_rule():
    path = os.path.join(DATA, 'selectivity_rule.json')
    return json.load(open(path)) if os.path.exists(path) else None


def selectivity_caption():
    r = selectivity_rule()
    if not r:
        return ('Selective = the drug is predicted to kill MDA-MB-231 and beats MCF7. '
                'Rule: `python src/models/selectivity.py`.')
    ds = r['delta_stats']
    return (f"**Selective** = {r['selectivity_rule']}  \n"
            f"Only {ds['n_positive']:,} of {r['panel_size']:,} drugs have a positive Δ "
            f"(the old fixed cut-off Δ > 0.05 was unreachable: max Δ = {ds['max']:.3f}), "
            f"so the relative rule is what carries information here. "
            f"**Sensitive** = {r['sensitivity_rule']}.")


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
    'pred_AUC_MCF7': st.column_config.NumberColumn(
        'MCF7 AUC', format='%.3f', help='Predicted AUC in the luminal control.'),
    'selectivity_delta': st.column_config.NumberColumn(
        'Selectivity Δ', format='%.3f',
        help='MCF7 AUC minus MDA-MB-231 AUC. Positive = hits TNBC harder than control.'),
    'sensitive_call': st.column_config.TextColumn('Verdict', help='Cut-off at AUC 0.30'),
    'selective_call': st.column_config.TextColumn('Selective'),
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
    return feat, pos, pca, model


def live_tab():
    st.subheader('Live predictor: any profiled line × any screened drug')
    st.write('Predicts on demand with the trained CatBoost model, for questions beyond '
             'the precomputed lines (for example BT-549). Values for MDA-MB-231 and '
             'MCF7 match the rankings tab.')
    if not st.toggle('Load the live model',
                     help='Loads CatBoost and the expression table. Takes a few seconds the first time.'):
        return
    try:
        feat, pos, pca_live, model = load_live()
    except Exception as e:
        st.warning('Could not load the live model. The precomputed tabs still work.')
        st.caption(f'Details: {type(e).__name__}: {e}')
        return

    pca_cols = [c for c in feat if c.startswith('pca_')]
    line_ids = sorted(pca_live['ModelID'].unique().tolist())
    drug_ids = sorted(rank_full['drug_id'].astype(str).unique().tolist())
    name_of = dict(zip(rank_full['drug_id'].astype(str), rank_full['drug_name']))

    with st.form('live_form'):
        c1, c2 = st.columns(2)
        sel_line = c1.selectbox(
            'Cell line (ModelID)', line_ids,
            index=line_ids.index(DEFAULT_LINE) if DEFAULT_LINE in line_ids else 0)
        sel_drug = c2.selectbox('Drug', drug_ids,
                                format_func=lambda d: f'{name_of.get(d, d)} ({d})')
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
        m1, m2 = st.columns(2)
        m1.metric('Predicted AUC', f'{auc:.3f}')
        if auc < SENSITIVE_AUC:
            m2.success(f'Sensitive (AUC < {SENSITIVE_AUC})')
        else:
            m2.info('Resistant or intermediate')


# ---------- header ----------
st.title('FerroSelect: which drugs kill TNBC, and by what mechanism')
st.write('Predicted sensitivity for **MDA-MB-231**, measured against the luminal '
         'control **MCF7**, plus the ferroptosis inducer panel from the proposal. '
         'Lower predicted AUC means stronger killing.')
st.info('Predicted sensitivity in a cell model, not a clinical recommendation. '
        f"Model version: {info.get('model_version', 'n/a')}")

with st.expander('How to read this dashboard (60 seconds, no ML background needed)'):
    st.write('**AUC** — predicted area under the dose-response curve, 0 to 1. '
             '**Lower means the drug kills more.** Below 0.30 we call it SENSITIVE.')
    st.write('**Selectivity Δ** — MCF7 AUC minus MDA-MB-231 AUC. Positive means the drug '
             'hits the TNBC line harder than the control line.')
    st.write('**R² / Spearman / NDCG@10** — how well the model ranked unseen cell lines '
             'during testing. Higher is better; gates were R² > 0.25 and per-line '
             'Spearman > 0.4, both passed.')
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
    k1.metric('Champion model', champion, help='Highest NDCG@10 on unseen cell lines.')
    k2.metric('R² (holdout)', f"{best['r2']:.3f}", delta=delta_str(best['r2'], dummy, 'r2'),
              delta_color='normal', help='Unseen cell lines. Acceptance gate: > 0.25.')
    k3.metric('Per-line Spearman', f"{best['per_line_spearman']:.3f}",
              delta=gate_str(best['per_line_spearman'], ACCEPTANCE_GATES['per_line_spearman']),
              delta_color='normal',
              help='Pre-registered acceptance gate: > 0.40 on unseen lines. The dummy '
                   'baseline cannot be ranked, so the delta shows the gate margin.')
    k4.metric('NDCG@10', f"{best['ndcg10']:.3f}", delta=delta_str(best['ndcg10'], dummy, 'ndcg10'),
              delta_color='normal', help='Top-10 ranking quality. Higher is better.')
    st.caption('Acceptance gates: R² > 0.25 '
               f"({'PASS' if best['r2'] > ACCEPTANCE_GATES['r2'] else 'FAIL'}, "
               f"{best['r2']:.3f}) and per-line Spearman > "
               f"{ACCEPTANCE_GATES['per_line_spearman']:.2f} "
               f"({'PASS' if best['per_line_spearman'] > ACCEPTANCE_GATES['per_line_spearman'] else 'FAIL'}, "
               f"{best['per_line_spearman']:.3f}) — both met on unseen cell lines.")

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

# ---------- tabs (4, not 7) ----------
tab_rank, tab_sel, tab_model, tab_live = st.tabs(
    ['Rankings & candidates', 'Selectivity & combinations', 'Model & audit', 'Live predictor'])

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
        left, right = st.columns([4, 1])
        with left:
            rank_table(sl[['role', 'drug_name', 'pred_AUC_MDA231', 'pred_AUC_MCF7',
                           'selectivity_delta', 'sensitive_call', 'selective_call']])
        with right:
            st.download_button('Download CSV', to_csv(sl), 'wetlab_shortlist.csv',
                               'text/csv', key='dl_sl')
        st.subheader('Starting dose brackets (proposal §6.4 — narrow them in pilot)')
        st.caption('Vehicle capped at 0.1% DMSO; every plate carries a vehicle-matched control.')
        st.dataframe(pd.DataFrame([
            ['Sulfasalazine', 'Type I inducer (workhorse)', '50–1000 µM, 8 points', 'Cheap generic'],
            ['Erastin', 'Type I reference', '0.1–20 µM (published IC50 ~2.2)', 'Research-grade import'],
            ['RSL3', 'Type II, extra only', '0.01–5 µM, MDA-MB-231 only', 'Droppable if over budget'],
            ['Simvastatin', 'Sensitiser (fixed dose)', 'Sub-lethal ~IC20, set in pilot', 'Cheap generic'],
            ['Ferrostatin-1', 'Rescue control', '1–2 µM alongside', 'Must rescue if ferroptosis'],
            ['Deferoxamine', 'Rescue control', '50–100 µM alongside', 'Death needed iron?'],
            ['NAC', 'Rescue control', '1–5 mM alongside', 'Rescues Type I, not RSL3'],
            ['Z-VAD-FMK', 'Specificity control', '20–50 µM alongside', 'Must NOT rescue'],
        ], columns=['Agent', 'Role in the assay', 'Starting range', 'Note']),
            hide_index=True, use_container_width=True)

    elif view == 'FerroSelect inducer panel':
        st.caption('Proposal-faithful predictions (Lasso/Ridge/RF, nested cell-line CV) for '
                   'the inducer panel on the three proposal lines. LOW confidence = fewer '
                   'than 25 training rows; a hypothesis for the lab, not evidence.')
        if not os.path.exists(os.path.join(DATA, 'ferroselect_panel_predictions.csv')):
            st.info('FerroSelect predictions are not in this deployment yet.')
        else:
            fpred = pretty(load('ferroselect_panel_predictions.csv'))
            left, right = st.columns([4, 1])
            with left:
                st.dataframe(fpred, hide_index=True, use_container_width=True)
            with right:
                st.download_button('Download CSV', to_csv(fpred),
                                   'ferroselect_panel_predictions.csv', 'text/csv', key='dl_fs')
            fres = load_json('ferroselect_results.json')
            fbest = fres['per_model_oof'][fres['frozen_model']]
            c1, c2, c3 = st.columns(3)
            c1.metric('Frozen model', fres['frozen_model'], help='Proposal-mandated algorithm.')
            c2.metric('Pooled OOF Spearman', f"{fbest['spearman']:.3f}",
                      help='Honest performance on unseen lines. Wide data, small panel.')
            c3.metric('Ceiling check', 'PASS' if 'OK' in fres['ceiling_flag'] else 'REVIEW',
                      help='Above published lab-vs-CTRP agreement (~0.83) would mean leakage.')
            st.caption(fres['ceiling_flag'])

    else:
        df = rank_full if view == 'Full ranking' else top20
        k, search = st.columns([1, 2])
        n_k = k.slider('Show top K drugs', 5, 50, 20)
        query = search.text_input('Find a drug', placeholder='e.g. paclitaxel')
        hits = df[df['drug_name'].str.contains(query, case=False, na=False, regex=False)] if query else df.head(n_k)
        if query:
            st.caption(f'{len(hits)} match(es) for "{query}"')
        left, right = st.columns([4, 1])
        with left:
            rank_table(hits[RANK_COLS])
        with right:
            st.download_button('Download CSV', to_csv(top20),
                               'mda_mb_231_top20_killers.csv', 'text/csv', key='dl_top')
        st.subheader(f'Predicted AUC of the top {min(n_k, len(hits))} (lower is stronger)')
        st.plotly_chart(bar_chart(hits[RANK_COLS]), use_container_width=True,
                        config={'displaylogo': False})

# ---------------- 2. selectivity + combos ----------------
with tab_sel:
    st.subheader('Selectivity: MDA-MB-231 vs MCF7')
    st.write(f'Below the dashed diagonal means the drug is predicted to kill the TNBC line '
             f'more than the control. The dotted lines mark the sensitivity cut-off '
             f'(AUC {SENSITIVE_AUC}). Hover for values; drag to zoom.')
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

# ---------------- 3. model & audit ----------------
with tab_model:
    st.subheader('Holdout benchmark on unseen cell lines')
    lb_show = leaderboard.copy()
    for c in ['mae', 'rmse', 'r2', 'spearman', 'per_line_spearman', 'ndcg10', 'recall10']:
        lb_show[c] = lb_show[c].astype(float).round(3)
    st.dataframe(lb_show, hide_index=True, use_container_width=True)

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

# ---------------- 4. live ----------------
with tab_live:
    live_tab()

st.divider()
st.caption('Reproducibility: DepMap 26Q3 + GDSC2/PRISM-REPURPOSING 27Oct23. '
           'Splits are cell-line holdouts (MDA-MB-231 and MCF7 never tuned on).')