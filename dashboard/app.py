"""TNBC Drug Response Predictor — Streamlit dashboard.

Run:  streamlit run dashboard/app.py
Reads all artifacts from ../outputs_local (no retraining, no API needed).
"""
import json
import os

import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'outputs_local'))

st.set_page_config(page_title='TNBC Drug Response Predictor', layout='wide')


@st.cache_data
def load(name):
    return pd.read_csv(os.path.join(DATA, name))


@st.cache_data
def load_info():
    with open(os.path.join(DATA, 'model_info.json')) as f:
        return json.load(f)


rank_full = load('mda_mb_231_full_ranking.csv')
top20 = load('mda_mb_231_top20_killers.csv')
shortlist = load('wetlab_shortlist.csv')
combos = load('combo_rankings.csv')
leaderboard = load('leaderboard.csv')
info = load_info()

# ---------- header ----------
st.title('TNBC Drug Response Predictor')
st.write('Predicted drug sensitivity for the triple-negative breast cancer model '
         '**MDA-MB-231**, with selectivity measured against the luminal control **MCF7**. '
         'Lower predicted AUC means higher sensitivity.')
st.caption('Predicted sensitivity in cell model — not a clinical recommendation. '
           f"Model version: {info.get('model_version', 'n/a')}")

# ---------- sidebar ----------
st.sidebar.header('Experiment summary')
st.sidebar.write(f"Training pairs: **30,994** (30 lines x 1,482 drugs, deduplicated)")
st.sidebar.write(f"Holdout lines: **7** (incl. MDA-MB-231, MCF7)")
st.sidebar.write(f"Champion: **CatBoost** (R2 0.567, per-line Spearman 0.630, NDCG@10 0.941)")
st.sidebar.divider()
st.sidebar.download_button('Download Top-20 CSV',
                           top20.to_csv(index=False), 'mda_mb_231_top20_killers.csv')
st.sidebar.download_button('Download wet-lab shortlist CSV',
                           shortlist.to_csv(index=False), 'wetlab_shortlist.csv')
st.sidebar.download_button('Download combo rankings CSV',
                           combos.to_csv(index=False), 'combo_rankings.csv')

# ---------- tabs ----------
tab_rank, tab_sel, tab_lab, tab_combo, tab_model = st.tabs(
    ['Killer ranking', 'Selectivity', 'Wet-lab shortlist', 'Combinations', 'Model'])

with tab_rank:
    st.subheader('Drugs ranked by predicted killing of MDA-MB-231')
    k = st.slider('Show top K drugs', 5, 50, 20)
    view = rank_full.head(k)[['rank', 'drug_name', 'pred_AUC_MDA231',
                              'pred_AUC_MCF7', 'selectivity_delta',
                              'sensitive_call', 'selective_call']]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.subheader('Predicted AUC of top candidates (lower is stronger)')
    chart = top20.head(k).set_index('drug_name')['pred_AUC_MDA231']
    st.bar_chart(chart, horizontal=True)

with tab_sel:
    st.subheader('Selectivity: MDA-MB-231 vs MCF7 predicted AUC')
    st.write('Points below the diagonal kill the TNBC line more than the control. '
             'Annotated labels mark the current Top-5.')
    sel = rank_full[['drug_name', 'pred_AUC_MDA231', 'pred_AUC_MCF7']].copy()
    st.scatter_chart(sel, x='pred_AUC_MCF7', y='pred_AUC_MDA231')
    st.dataframe(top20[['rank', 'drug_name', 'pred_AUC_MDA231',
                        'pred_AUC_MCF7', 'selectivity_delta']],
                 use_container_width=True, hide_index=True)

with tab_lab:
    st.subheader('Handoff for MTT validation (72h dose-response, both lines)')
    st.write('Top-5 AI candidates plus Doxorubicin / Paclitaxel positive controls '
             'and one predicted-resistant compound (bottom row).')
    st.dataframe(shortlist[['drug_name', 'pred_AUC_MDA231', 'pred_AUC_MCF7',
                            'selectivity_delta']], use_container_width=True, hide_index=True)

with tab_combo:
    st.subheader('Two-drug combinations (Bliss independence)')
    st.write('Ranked by selectivity gap: expected efficacy on MDA-MB-231 minus '
             'expected efficacy on MCF7. Additivity assumption — synergy verdict '
             'requires 4x4 checkerboard validation.')
    st.dataframe(combos, use_container_width=True, hide_index=True)

with tab_model:
    st.subheader('Holdout benchmark (unseen cell lines)')
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    best = leaderboard.sort_values('ndcg10', ascending=False).iloc[0]
    c1.metric('Best model', best['label'])
    c2.metric('R2 (gate > 0.25)', f"{best['r2']:.3f}")
    c3.metric('Per-line Spearman (gate > 0.4)', f"{best['per_line_spearman']:.3f}")
    st.subheader('SHAP explanations')
    col_a, col_b = st.columns(2)
    with col_a:
        st.write('Global feature impact (Top-20)')
        st.image(os.path.join(DATA, 'shap_global_top20.png'), use_container_width=True)
    with col_b:
        st.write('Local explanation: MDA-MB-231 x top killer')
        st.image(os.path.join(DATA, 'shap_local_mda231_top1.png'), use_container_width=True)

st.divider()
st.caption('Reproducibility: DepMap 26Q3 + GDSC2/PRISM-REPURPOSING 27Oct23. '
           'Splits are cell-line holdouts (MDA-MB-231, MCF7 never tuned on). '
           'See model_info.json and audit_summary.json for versions.')
