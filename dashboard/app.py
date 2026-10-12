"""TNBC Drug Response Predictor — Streamlit dashboard.

Run:  streamlit run dashboard/app.py
Needs Streamlit >= 1.50 (uses width="stretch"; use_container_width is deprecated).
Reads all artifacts from ../outputs_local (no retraining, no API needed).
"""
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'outputs_local'))

SENSITIVE_AUC = 0.3                     # same cut-off as sensitive_call
HIGHLIGHT, MUTED = '#D9480F', '#868E96'  # readable on light and dark themes
TOP_LABEL, OTHER_LABEL = 'Top candidates', 'Other drugs'
POSITIVE_CONTROLS = {'doxorubicin', 'paclitaxel'}
DEFAULT_LINE = 'ACH-000768'             # MDA-MB-231
CSVS = ['mda_mb_231_full_ranking.csv', 'mda_mb_231_top20_killers.csv',
        'wetlab_shortlist.csv', 'combo_rankings.csv', 'leaderboard.csv']

st.set_page_config(page_title='TNBC Drug Response Predictor',
                   page_icon='🧬', layout='wide')

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
def load_info():
    with open(os.path.join(DATA, 'model_info.json')) as f:
        return json.load(f)


@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


rank_full = load('mda_mb_231_full_ranking.csv')
top20 = load('mda_mb_231_top20_killers.csv')
shortlist = load('wetlab_shortlist.csv')
combos = load('combo_rankings.csv')
leaderboard = load('leaderboard.csv')
info = load_info()
best = leaderboard.sort_values('ndcg10', ascending=False).iloc[0]

# ---------- shared table config ----------
COLS = {
    'rank': st.column_config.NumberColumn('Rank', format='%d', width='small'),
    'role': st.column_config.TextColumn('Role'),
    'drug_name': st.column_config.TextColumn('Drug'),
    'pred_AUC_MDA231': st.column_config.NumberColumn(
        'MDA-MB-231 AUC', format='%.3f', help='Predicted AUC. Lower means more sensitive.'),
    'pred_AUC_MCF7': st.column_config.NumberColumn(
        'MCF7 AUC', format='%.3f', help='Predicted AUC in the luminal control.'),
    'selectivity_delta': st.column_config.NumberColumn(
        'Selectivity Δ', format='%.3f', help='Predicted-AUC gap between MDA-MB-231 and MCF7.'),
    'sensitive_call': st.column_config.Column('Sensitive'),
    'selective_call': st.column_config.Column('Selective'),
}


def table(df):
    st.dataframe(df, hide_index=True, width='stretch',
                 column_config={c: COLS[c] for c in df.columns if c in COLS})


# ---------- charts ----------
def top_chart(df):
    """Horizontal bars in rank order (strongest killer at the top)."""
    return (alt.Chart(df).mark_bar().encode(
        x=alt.X('pred_AUC_MDA231:Q', title='Predicted AUC (lower = stronger)'),
        y=alt.Y('drug_name:N', sort=None, title=None),
        color=alt.Color('selectivity_delta:Q', title='Selectivity Δ',
                        scale=alt.Scale(scheme='viridis')),
        tooltip=[alt.Tooltip('drug_name:N', title='Drug'),
                 alt.Tooltip('pred_AUC_MDA231:Q', title='MDA-MB-231 AUC', format='.3f'),
                 alt.Tooltip('pred_AUC_MCF7:Q', title='MCF7 AUC', format='.3f'),
                 alt.Tooltip('selectivity_delta:Q', title='Selectivity Δ', format='.3f')],
    ).properties(height=max(140, 24 * len(df))))


def selectivity_chart(df, n_top):
    """Scatter with y=x diagonal, sensitivity cut-off and labelled top-N."""
    d = df.copy()
    is_top = d['rank'] <= n_top
    d['group'] = np.where(is_top, TOP_LABEL, OTHER_LABEL)
    d['_alpha'] = np.where(is_top, 1.0, 0.35)
    d['_size'] = np.where(is_top, 90, 30)

    lo = float(min(d['pred_AUC_MCF7'].min(), d['pred_AUC_MDA231'].min()))
    hi = float(max(d['pred_AUC_MCF7'].max(), d['pred_AUC_MDA231'].max()))
    pad = 0.02 * (hi - lo)
    dom = alt.Scale(domain=[lo - pad, hi + pad])
    x = alt.X('pred_AUC_MCF7:Q', title='Predicted AUC, MCF7 (control)', scale=dom)
    y = alt.Y('pred_AUC_MDA231:Q', title='Predicted AUC, MDA-MB-231 (TNBC)', scale=dom)

    diag = pd.DataFrame({'pred_AUC_MCF7': [lo, hi], 'pred_AUC_MDA231': [lo, hi]})
    cut = pd.DataFrame({'pred_AUC_MDA231': [SENSITIVE_AUC]})
    line = alt.Chart(diag).mark_line(color=MUTED, strokeDash=[5, 4]).encode(x=x, y=y)
    rule = alt.Chart(cut).mark_rule(color=MUTED, strokeDash=[2, 4]).encode(y=y)
    pts = alt.Chart(d).mark_circle().encode(
        x=x, y=y,
        color=alt.Color('group:N',
                        scale=alt.Scale(domain=[TOP_LABEL, OTHER_LABEL],
                                        range=[HIGHLIGHT, MUTED]),
                        legend=alt.Legend(title=None, orient='bottom')),
        opacity=alt.Opacity('_alpha:Q', scale=None, legend=None),
        size=alt.Size('_size:Q', scale=None, legend=None),
        tooltip=[alt.Tooltip('drug_name:N', title='Drug'),
                 alt.Tooltip('pred_AUC_MDA231:Q', title='MDA-MB-231 AUC', format='.3f'),
                 alt.Tooltip('pred_AUC_MCF7:Q', title='MCF7 AUC', format='.3f')])
    labels = (alt.Chart(d[is_top]).mark_text(align='left', dx=8, fontSize=12, color=HIGHLIGHT)
              .encode(x=x, y=y, text='drug_name:N'))
    return alt.layer(line, rule, pts, labels).properties(height=460).interactive()


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
             'the two precomputed lines (for example BT-549). Values for MDA-MB-231 and '
             'MCF7 match the ranking tab.')
    if not st.toggle('Load the live model',
                     help='Loads CatBoost and the expression table. Takes a few seconds the first time.'):
        return
    try:
        feat, pos, pca_live, model = load_live()
    except Exception as e:  # show the real cause, not a guess
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
        m1.metric('Predicted AUC', f'{auc:.4f}')
        if auc < SENSITIVE_AUC:
            m2.success(f'Sensitive (AUC < {SENSITIVE_AUC})')
        else:
            m2.info('Resistant or intermediate')


# ---------- header ----------
st.title('TNBC Drug Response Predictor')
st.write('Predicted drug sensitivity for the triple-negative breast cancer model '
         '**MDA-MB-231**, with selectivity measured against the luminal control **MCF7**. '
         'Lower predicted AUC means higher sensitivity.')
st.info('Predicted sensitivity in a cell model, not a clinical recommendation. '
        f"Model version: {info.get('model_version', 'n/a')}", icon='⚠️')

with st.container(border=True):
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Champion model', best['label'])
    k2.metric('R²', f"{best['r2']:.3f}", help='Unseen cell lines. Acceptance gate: > 0.25')
    k3.metric('Per-line Spearman', f"{best['per_line_spearman']:.3f}",
              help='Acceptance gate: > 0.4')
    k4.metric('NDCG@10', f"{best['ndcg10']:.3f}")

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

# ---------- tabs ----------
tab_rank, tab_sel, tab_lab, tab_combo, tab_model, tab_live = st.tabs(
    ['Killer ranking', 'Selectivity', 'Wet-lab shortlist', 'Combinations',
     'Model', 'Live predictor'])

with tab_rank:
    st.subheader('Drugs ranked by predicted killing of MDA-MB-231')
    c1, c2 = st.columns([1, 2])
    k = c1.slider('Show top K drugs', 5, 50, 20)
    query = c2.text_input('Find a drug in the full ranking', placeholder='e.g. paclitaxel')
    cols = ['rank', 'drug_name', 'pred_AUC_MDA231', 'pred_AUC_MCF7',
            'selectivity_delta', 'sensitive_call', 'selective_call']
    if query:
        hits = rank_full[rank_full['drug_name'].str.contains(
            query, case=False, na=False, regex=False)]
        st.caption(f'{len(hits)} match(es) for "{query}"')
        table(hits[cols])
    else:
        table(rank_full.head(k)[cols])
    st.download_button('Download Top-20 CSV', to_csv(top20),
                       'mda_mb_231_top20_killers.csv', 'text/csv')
    st.subheader(f'Predicted AUC of the top {k} (lower is stronger)')
    st.altair_chart(top_chart(rank_full.head(k)), width='stretch')

with tab_sel:
    st.subheader('Selectivity: MDA-MB-231 vs MCF7 predicted AUC')
    st.write('Points below the dashed diagonal are predicted to kill the TNBC line more than '
             f'the control. The dotted line marks the sensitivity cut-off (AUC {SENSITIVE_AUC}). '
             'Hover for values; scroll to zoom.')
    n_top = st.slider('Label the top N drugs', 0, 10, 5)
    st.altair_chart(selectivity_chart(rank_full, n_top), width='stretch')
    table(top20[['rank', 'drug_name', 'pred_AUC_MDA231', 'pred_AUC_MCF7', 'selectivity_delta']])

with tab_lab:
    st.subheader('Handoff for MTT validation (72 h dose-response, both lines)')
    st.write('Top-5 AI candidates, Doxorubicin and Paclitaxel as positive controls, '
             'and one predicted-resistant compound as the negative control.')
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
    table(sl[['role', 'drug_name', 'pred_AUC_MDA231', 'pred_AUC_MCF7', 'selectivity_delta']])
    st.download_button('Download wet-lab shortlist CSV', to_csv(shortlist),
                       'wetlab_shortlist.csv', 'text/csv')

with tab_combo:
    st.subheader('Two-drug combinations (Bliss independence)')
    st.write('Ranked by selectivity gap: expected efficacy on MDA-MB-231 minus expected '
             'efficacy on MCF7. This assumes additivity, so a synergy verdict needs 4×4 '
             'checkerboard validation.')
    table(combos)
    st.download_button('Download combo rankings CSV', to_csv(combos),
                       'combo_rankings.csv', 'text/csv')

with tab_model:
    st.subheader('Holdout benchmark (unseen cell lines)')
    table(leaderboard)
    st.subheader('SHAP explanations')
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
                st.image(path, caption=alt_text, width='stretch')
            else:
                st.caption(f'`{fname}` not found in the outputs folder.')

with tab_live:
    live_tab()

st.divider()
st.caption('Reproducibility: DepMap 26Q3 + GDSC2/PRISM-REPURPOSING 27Oct23. '
           'Splits are cell-line holdouts (MDA-MB-231, MCF7 never tuned on).')
