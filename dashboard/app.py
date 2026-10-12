"""TNBC Drug Response Predictor — polished Streamlit dashboard.

Run from the project root with:
    streamlit run dashboard/app.py

All precomputed artifacts are read from ../outputs_local. The dashboard does not
retrain the model or require an API. The live predictor loads CatBoost only when
its tab is opened.
"""

from __future__ import annotations

import json
import os
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "outputs_local"))

st.set_page_config(
    page_title="TNBC Drug Response Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- visual system ----------
st.markdown(
    """
    <style>
    :root {
      --ink: #142235;
      --muted: #64748b;
      --teal: #0f766e;
      --teal-soft: #e8f5f2;
      --line: #e2e8f0;
      --surface: #ffffff;
    }
    .main .block-container {
      max-width: 1600px;
      padding-top: 1.6rem;
      padding-bottom: 2.5rem;
    }
    [data-testid="stAppViewContainer"] { background: #f7f9fc; }
    [data-testid="stHeader"] { background: rgba(247, 249, 252, 0.88); }
    [data-testid="stSidebar"] {
      background: #f1f5f9;
      border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    h1 { font-weight: 760 !important; }
    [data-testid="stMetric"] {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 1rem 1.1rem;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.025);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-baseweb="tab-list"] {
      gap: 0.35rem;
      border-bottom: 1px solid var(--line);
    }
    [data-baseweb="tab"] {
      height: 3rem;
      padding: 0 1rem;
      color: #475569;
      border-radius: 9px 9px 0 0;
    }
    [aria-selected="true"][data-baseweb="tab"] {
      color: var(--teal) !important;
      background: var(--teal-soft);
    }
    div[data-testid="stDataFrame"] {
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: white;
    }
    div.stButton > button, div.stDownloadButton > button {
      border-radius: 9px;
      font-weight: 600;
    }
    .tnbc-hero {
      background: linear-gradient(115deg, #10243a 0%, #173c52 62%, #17665f 100%);
      color: #f8fafc;
      border-radius: 18px;
      padding: 1.55rem 1.8rem 1.45rem 1.8rem;
      margin-bottom: 1rem;
      border: 1px solid rgba(255,255,255,0.08);
    }
    .tnbc-eyebrow {
      font-size: 0.72rem;
      font-weight: 750;
      letter-spacing: 0.13em;
      text-transform: uppercase;
      color: #a9dfd6;
      margin-bottom: 0.55rem;
    }
    .tnbc-title {
      font-size: clamp(1.65rem, 3vw, 2.35rem);
      line-height: 1.15;
      font-weight: 780;
      letter-spacing: -0.04em;
      color: #ffffff;
      margin-bottom: 0.6rem;
    }
    .tnbc-subtitle {
      max-width: 900px;
      color: #d9e7ee;
      font-size: 0.98rem;
      line-height: 1.55;
    }
    .section-note {
      color: #64748b;
      font-size: 0.9rem;
      line-height: 1.55;
    }
    .mini-label {
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.72rem;
      font-weight: 700;
    }
    @media (max-width: 768px) {
      .main .block-container { padding: 1rem 0.8rem 2rem 0.8rem; }
      .tnbc-hero { padding: 1.25rem; border-radius: 13px; }
      [data-baseweb="tab"] { padding: 0 0.55rem; font-size: 0.82rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- data access ----------
@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(DATA, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_info() -> dict[str, Any]:
    path = os.path.join(DATA, "model_info.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        parsed = json.load(handle)
    return parsed if isinstance(parsed, dict) else {}


@st.cache_resource(show_spinner="Loading the live CatBoost predictor…")
def load_live_artifacts():
    """Load the model and expression matrix only when the Live tab is opened."""
    import pyarrow.parquet as pq
    from catboost import CatBoostRegressor

    parquet_path = os.path.join(DATA, "tnbc_ml_dataset.parquet")
    pca_path = os.path.join(DATA, "expression_pca128_breast.parquet")
    model_path = os.path.join(DATA, "catboost_tnbc_v1.cbm")
    for path in (parquet_path, pca_path, model_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    columns = pq.read_schema(parquet_path).names
    features = [
        column
        for column in columns
        if column.startswith("pca_")
        or (column.startswith("drug_") and column not in ("drug_id", "drug_name"))
    ]
    if not features:
        raise ValueError("No PCA/drug features were found in tnbc_ml_dataset.parquet.")

    positions = {column: index for index, column in enumerate(features)}
    pca = pd.read_parquet(pca_path)
    if "ModelID" not in pca.columns:
        raise ValueError("The expression PCA file is missing its ModelID column.")
    pca = pca.drop_duplicates(subset=["ModelID"], keep="first").reset_index(drop=True)

    model = CatBoostRegressor()
    model.load_model(model_path)
    return features, positions, pca, model


def ordered_by_rank(frame: pd.DataFrame) -> pd.DataFrame:
    if "rank" in frame.columns:
        return frame.sort_values("rank", ascending=True, kind="stable")
    return frame.copy()


def auc_column_config(frame: pd.DataFrame) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for column in frame.columns:
        if "auc" in column.lower() or "delta" in column.lower():
            configs[column] = st.column_config.NumberColumn(column, format="%.4f")
        elif column.lower() == "rank":
            configs[column] = st.column_config.NumberColumn(column, format="%d")
        elif column.lower() in {"ndcg10", "r2", "per_line_spearman", "spearman"}:
            configs[column] = st.column_config.NumberColumn(column, format="%.3f")
    return configs


def show_table(frame: pd.DataFrame, *, height: int = 420, key: str | None = None) -> None:
    st.dataframe(
        frame,
        width="stretch",
        height=height,
        hide_index=True,
        column_config=auc_column_config(frame),
        key=key,
    )


def as_positive_count(series: pd.Series) -> int:
    normalized = series.astype(str).str.strip().str.lower()
    return int(normalized.isin({"true", "1", "yes", "selective", "sensitive"}).sum())


# Load the lightweight, precomputed tables first. If a required artifact is
# absent, show the expected path instead of a long, opaque traceback.
try:
    rank_full = load_csv("mda_mb_231_full_ranking.csv")
    top20 = load_csv("mda_mb_231_top20_killers.csv")
    shortlist = load_csv("wetlab_shortlist.csv")
    combos = load_csv("combo_rankings.csv")
    leaderboard = load_csv("leaderboard.csv")
    info = load_info()
except Exception as exc:
    st.error("The dashboard could not load its precomputed artifacts.")
    st.write("Confirm these files exist in the project’s `outputs_local/` directory.")
    st.code(str(exc))
    st.stop()

required_ranking_columns = {
    "drug_name", "pred_AUC_MDA231", "pred_AUC_MCF7", "selectivity_delta"
}
missing_columns = sorted(required_ranking_columns - set(rank_full.columns))
if missing_columns:
    st.error("The ranking CSV is missing required columns: " + ", ".join(missing_columns))
    st.stop()

ranked = ordered_by_rank(rank_full)
model_version = str(info.get("model_version", "n/a"))

# Use the supplied leaderboard for the model summary when possible, so the
# sidebar cannot silently drift away from the actual benchmark artifact.
model_label_column = next(
    (column for column in ("label", "model", "model_name") if column in leaderboard.columns),
    leaderboard.columns[0] if len(leaderboard.columns) else None,
)
if not leaderboard.empty:
    if model_label_column is not None:
        catboost_rows = leaderboard[
            leaderboard[model_label_column].astype(str).str.contains("catboost", case=False, na=False)
        ]
    else:
        catboost_rows = leaderboard.iloc[0:0]
    summary_row = (
        catboost_rows.iloc[0]
        if not catboost_rows.empty
        else (leaderboard.sort_values("ndcg10", ascending=False).iloc[0] if "ndcg10" in leaderboard.columns else leaderboard.iloc[0])
    )
    summary_model_name = str(summary_row.get(model_label_column, "n/a")) if model_label_column else "n/a"
    summary_r2 = f"{float(summary_row['r2']):.3f}" if "r2" in leaderboard.columns and pd.notna(summary_row.get("r2")) else "n/a"
    summary_spearman_column = "per_line_spearman" if "per_line_spearman" in leaderboard.columns else "spearman"
    summary_spearman = f"{float(summary_row[summary_spearman_column]):.3f}" if summary_spearman_column in leaderboard.columns and pd.notna(summary_row.get(summary_spearman_column)) else "n/a"
    summary_ndcg = f"{float(summary_row['ndcg10']):.3f}" if "ndcg10" in leaderboard.columns and pd.notna(summary_row.get("ndcg10")) else "n/a"
else:
    summary_model_name = "Unavailable"
    summary_r2 = summary_spearman = summary_ndcg = "n/a"

# ---------- sidebar ----------
st.sidebar.markdown("## Experiment summary")
st.sidebar.caption("Precomputed research results")
st.sidebar.metric("Training pairs", "30,994")
st.sidebar.metric("Holdout cell lines", "7")
st.sidebar.markdown(
    f"**Benchmark model**  \n{summary_model_name}  \n"
    f"R² {summary_r2} · Per-line Spearman {summary_spearman} · NDCG@10 {summary_ndcg}"
)
st.sidebar.caption(f"Model version: {model_version}")
st.sidebar.divider()
st.sidebar.markdown("### Export data")
st.sidebar.download_button(
    "Download top-20 ranking",
    data=top20.to_csv(index=False).encode("utf-8"),
    file_name="mda_mb_231_top20_killers.csv",
    mime="text/csv",
    use_container_width=True,
    key="download_top20_sidebar",
)
st.sidebar.download_button(
    "Download wet-lab shortlist",
    data=shortlist.to_csv(index=False).encode("utf-8"),
    file_name="wetlab_shortlist.csv",
    mime="text/csv",
    use_container_width=True,
    key="download_shortlist_sidebar",
)
st.sidebar.download_button(
    "Download combination rankings",
    data=combos.to_csv(index=False).encode("utf-8"),
    file_name="combo_rankings.csv",
    mime="text/csv",
    use_container_width=True,
    key="download_combos_sidebar",
)
st.sidebar.divider()
st.sidebar.caption(
    "Research-use predictions only. Cell-line sensitivity does not establish patient benefit."
)

# ---------- dashboard header ----------
st.markdown(
    f"""
    <div class="tnbc-hero">
      <div class="tnbc-eyebrow">Translational oncology · Research workspace</div>
      <div class="tnbc-title">TNBC Drug Response Predictor</div>
      <div class="tnbc-subtitle">
        A ranked, model-based view of predicted drug sensitivity in the
        <b>MDA-MB-231</b> triple-negative breast cancer cell line, with predicted
        selectivity assessed against the <b>MCF7</b> luminal control.
        Lower predicted AUC indicates higher predicted sensitivity.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    f"Model version: {model_version} · Precomputed outputs · No retraining or external API required"
)
st.warning(
    "Research-use only. These are computational predictions for cell models, not a clinical recommendation. "
    "Experimental validation is required before drawing biological conclusions.",
    icon="⚠️",
)

# Key figures are derived from the loaded artifacts, except the training/holdout
# summary above, which is the experiment metadata supplied in the original app.
metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Ranked compounds", f"{len(rank_full):,}")
metric_2.metric("Wet-lab shortlist", f"{len(shortlist):,}")
metric_3.metric("Ranked combinations", f"{len(combos):,}")
if "selective_call" in rank_full.columns:
    metric_4.metric("Predicted selective calls", f"{as_positive_count(rank_full['selective_call']):,}")
else:
    metric_4.metric("Model version", model_version)

st.markdown(" ")

tab_rank, tab_sel, tab_lab, tab_combo, tab_model, tab_live = st.tabs(
    ["Killer ranking", "Selectivity", "Wet-lab shortlist", "Combinations", "Model audit", "Live predictor"],
    on_change="rerun",
    key="tnbc_main_tabs",
)

# `on_change="rerun"` tracks the selected tab. Rendering only the open tab keeps
# the costly model/parquet load out of visits to the precomputed-results tabs.
if tab_rank.open:
    with tab_rank:
        st.subheader("Drug ranking")
        st.markdown(
            '<div class="section-note">Ranked by predicted response in MDA-MB-231. '
            'AUC below 0.30 is treated as a project-level sensitivity threshold.</div>',
            unsafe_allow_html=True,
        )

        controls = st.columns([1.5, 1, 1])
        with controls[0]:
            query = st.text_input("Search compound", placeholder="Enter a drug name…", key="ranking_search")
        with controls[1]:
            call_options = ["All"]
            if "sensitive_call" in ranked.columns:
                call_options += sorted(ranked["sensitive_call"].dropna().astype(str).unique().tolist())
            sensitivity_filter = st.selectbox("Sensitivity call", call_options, key="sensitivity_filter")
        with controls[2]:
            max_rows = max(1, min(50, len(ranked)))
            default_rows = min(20, max_rows)
            show_k = st.slider("Candidates to show", min_value=1, max_value=max_rows, value=default_rows, key="ranking_k")

        filtered = ranked.copy()
        if query.strip():
            filtered = filtered[
                filtered["drug_name"].astype(str).str.contains(query.strip(), case=False, regex=False, na=False)
            ]
        if sensitivity_filter != "All" and "sensitive_call" in filtered.columns:
            filtered = filtered[filtered["sensitive_call"].astype(str) == sensitivity_filter]
        visible = filtered.head(show_k)

        if visible.empty:
            st.info("No compounds match the current filters. Clear the search or select ‘All’ calls.")
        else:
            left, right = st.columns([1.15, 1])
            with left:
                st.markdown("#### Candidate table")
                display_columns = [
                    column for column in [
                        "rank", "drug_name", "pred_AUC_MDA231", "pred_AUC_MCF7",
                        "selectivity_delta", "sensitive_call", "selective_call"
                    ] if column in visible.columns
                ]
                show_table(visible[display_columns], height=min(560, 42 + 36 * len(visible)), key="ranking_table")
            with right:
                st.markdown("#### Predicted AUC")
                st.caption("Lower values indicate stronger predicted sensitivity; teal bars are below AUC 0.30.")
                chart_data = visible.copy()
                bars = (
                    alt.Chart(chart_data)
                    .mark_bar(cornerRadiusEnd=5, height=19)
                    .encode(
                        x=alt.X(
                            "pred_AUC_MDA231:Q",
                            title="Predicted AUC · MDA-MB-231",
                            axis=alt.Axis(format=".2f"),
                            scale=alt.Scale(zero=True),
                        ),
                        y=alt.Y(
                            "drug_name:N",
                            title=None,
                            sort=alt.SortField(field="pred_AUC_MDA231", order="ascending"),
                        ),
                        color=alt.condition(
                            alt.datum.pred_AUC_MDA231 < 0.3,
                            alt.value("#0f766e"),
                            alt.value("#9aabbc"),
                        ),
                        tooltip=[
                            alt.Tooltip("drug_name:N", title="Drug"),
                            alt.Tooltip("pred_AUC_MDA231:Q", title="Predicted AUC", format=".4f"),
                        ],
                    )
                    .properties(height=max(270, 27 * len(chart_data)), background="white")
                    .configure_view(stroke=None)
                    .configure_axis(gridColor="#e7edf3", labelColor="#475569", titleColor="#334155")
                )
                st.altair_chart(bars, width="stretch")

        st.download_button(
            "Export filtered ranking as CSV",
            data=visible.to_csv(index=False).encode("utf-8"),
            file_name="filtered_tnbc_drug_ranking.csv",
            mime="text/csv",
            disabled=visible.empty,
            key="download_filtered_ranking",
        )

if tab_sel.open:
    with tab_sel:
        st.subheader("Selectivity against the MCF7 control")
        st.markdown(
            '<div class="section-note">Each point represents a compound. Points below the diagonal '
            'have a lower predicted AUC in MDA-MB-231 than in MCF7, suggesting greater predicted '
            'sensitivity in the TNBC cell model. This is a computational comparison, not proof of selectivity.</div>',
            unsafe_allow_html=True,
        )
        selectivity = ranked[["drug_name", "pred_AUC_MDA231", "pred_AUC_MCF7", "selectivity_delta"]].copy()
        top_five = ranked.head(5)[["drug_name", "pred_AUC_MDA231", "pred_AUC_MCF7", "selectivity_delta"]].copy()
        all_values = pd.concat(
            [selectivity["pred_AUC_MDA231"], selectivity["pred_AUC_MCF7"]], ignore_index=True
        ).dropna()
        if selectivity.empty or all_values.empty:
            st.info("No valid selectivity records are available in the ranking artifact.")
        else:
            axis_min = float(all_values.min())
            axis_max = float(all_values.max())
            if axis_min == axis_max:
                axis_min -= 0.01
                axis_max += 0.01
            axis_domain = [axis_min, axis_max]
            points = (
                alt.Chart(selectivity)
                .mark_circle(size=54, opacity=0.58, color="#9aaec0")
                .encode(
                    x=alt.X("pred_AUC_MCF7:Q", title="MCF7 predicted AUC", scale=alt.Scale(domain=axis_domain)),
                    y=alt.Y("pred_AUC_MDA231:Q", title="MDA-MB-231 predicted AUC", scale=alt.Scale(domain=axis_domain)),
                    tooltip=[
                        alt.Tooltip("drug_name:N", title="Drug"),
                        alt.Tooltip("pred_AUC_MDA231:Q", title="MDA-MB-231 AUC", format=".4f"),
                        alt.Tooltip("pred_AUC_MCF7:Q", title="MCF7 AUC", format=".4f"),
                        alt.Tooltip("selectivity_delta:Q", title="Selectivity delta", format=".4f"),
                    ],
                )
            )
            diagonal_data = pd.DataFrame({"auc": axis_domain})
            diagonal = (
                alt.Chart(diagonal_data)
                .mark_line(color="#64748b", strokeDash=[5, 4], opacity=0.8)
                .encode(x="auc:Q", y="auc:Q")
            )
            highlights = (
                alt.Chart(top_five)
                .mark_circle(size=115, color="#0f766e", stroke="#ffffff", strokeWidth=1.5)
                .encode(
                    x=alt.X("pred_AUC_MCF7:Q", scale=alt.Scale(domain=axis_domain)),
                    y=alt.Y("pred_AUC_MDA231:Q", scale=alt.Scale(domain=axis_domain)),
                    tooltip=["drug_name:N", "pred_AUC_MDA231:Q", "pred_AUC_MCF7:Q", "selectivity_delta:Q"],
                )
            )
            labels = (
                alt.Chart(top_five)
                .mark_text(align="left", dx=7, dy=-7, fontSize=10, color="#115e59")
                .encode(
                    x=alt.X("pred_AUC_MCF7:Q", scale=alt.Scale(domain=axis_domain)),
                    y=alt.Y("pred_AUC_MDA231:Q", scale=alt.Scale(domain=axis_domain)),
                    text="drug_name:N",
                )
            )
            scatter = (
                alt.layer(diagonal, points, highlights, labels)
                .properties(height=470, background="white")
                .configure_view(stroke=None)
                .configure_axis(gridColor="#e7edf3", labelColor="#475569", titleColor="#334155")
            )
            st.altair_chart(scatter, width="stretch")
            st.markdown("#### Top-five candidates")
            show_table(top_five, height=255, key="selectivity_top5_table")
            st.download_button(
                "Export selectivity table",
                data=selectivity.to_csv(index=False).encode("utf-8"),
                file_name="tnbc_selectivity_comparison.csv",
                mime="text/csv",
                key="download_selectivity",
            )

if tab_lab.open:
    with tab_lab:
        st.subheader("Wet-lab validation shortlist")
        st.markdown(
            '<div class="section-note">Proposed handoff for 72-hour MTT dose-response validation in both '
            'cell lines. The list includes computational candidates and control compounds; use the project '
            'protocol to confirm concentration ranges, replicates, and assay controls.</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{len(shortlist):,} compounds in the supplied shortlist")
        show_table(shortlist, height=min(620, 48 + 38 * max(1, len(shortlist))), key="wetlab_table")
        st.download_button(
            "Download wet-lab handoff CSV",
            data=shortlist.to_csv(index=False).encode("utf-8"),
            file_name="wetlab_shortlist.csv",
            mime="text/csv",
            key="download_shortlist_main",
        )
        st.info(
            "Predicted sensitivity should be treated as a prioritization signal. It does not establish drug efficacy, "
            "selectivity, dose, safety, or clinical usefulness."
        )

if tab_combo.open:
    with tab_combo:
        st.subheader("Two-drug combination rankings")
        st.markdown(
            '<div class="section-note">The provided rankings use a Bliss-independence assumption. '
            'A high ranking is a prioritization hypothesis—not a demonstrated synergy result. '
            'A checkerboard experiment and an appropriate response-surface analysis are needed for validation.</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{len(combos):,} ranked combinations")
        if combos.empty:
            st.info("The combination ranking file contains no rows.")
        else:
            show_table(combos, height=540, key="combo_table")
        st.download_button(
            "Download combination rankings",
            data=combos.to_csv(index=False).encode("utf-8"),
            file_name="combo_rankings.csv",
            mime="text/csv",
            key="download_combos_main",
        )

if tab_model.open:
    with tab_model:
        st.subheader("Model audit and holdout performance")
        st.markdown(
            '<div class="section-note">Benchmark metrics are calculated on held-out cell lines. '
            'Compare models using the same split protocol; aggregate metrics do not guarantee reliable '
            'ranking for every cell line or compound.</div>',
            unsafe_allow_html=True,
        )
        if leaderboard.empty:
            st.info("The leaderboard artifact contains no rows.")
        else:
            best = leaderboard.sort_values("ndcg10", ascending=False).iloc[0] if "ndcg10" in leaderboard.columns else leaderboard.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            label_col = "label" if "label" in leaderboard.columns else leaderboard.columns[0]
            m1.metric("Top-ranked model", str(best.get(label_col, "n/a")))
            m2.metric("R²", f"{float(best['r2']):.3f}" if "r2" in leaderboard.columns and pd.notna(best.get("r2")) else "n/a")
            spearman_col = "per_line_spearman" if "per_line_spearman" in leaderboard.columns else "spearman"
            m3.metric("Per-line Spearman", f"{float(best[spearman_col]):.3f}" if spearman_col in leaderboard.columns and pd.notna(best.get(spearman_col)) else "n/a")
            m4.metric("NDCG@10", f"{float(best['ndcg10']):.3f}" if "ndcg10" in leaderboard.columns and pd.notna(best.get("ndcg10")) else "n/a")
            st.markdown("#### Holdout leaderboard")
            show_table(leaderboard, height=350, key="leaderboard_table")

        st.markdown("#### SHAP explanations")
        shap_left, shap_right = st.columns(2)
        shap_specs = [
            (shap_left, "Global feature impact · Top 20", "shap_global_top20.png"),
            (shap_right, "Local explanation · MDA-MB-231 top candidate", "shap_local_mda231_top1.png"),
        ]
        for column, title, filename in shap_specs:
            with column:
                st.markdown(f"**{title}**")
                path = os.path.join(DATA, filename)
                if os.path.isfile(path):
                    st.image(path, width="stretch")
                else:
                    st.info(f"Explanation image not found: `{filename}`")

        info_path = os.path.join(DATA, "audit_summary.json")
        if os.path.isfile(info_path):
            with st.expander("Reproducibility metadata"):
                try:
                    with open(info_path, encoding="utf-8") as handle:
                        st.json(json.load(handle))
                except (OSError, json.JSONDecodeError) as exc:
                    st.warning(f"Could not read audit_summary.json: {exc}")

if tab_live.open:
    with tab_live:
        st.subheader("Live sensitivity predictor")
        st.markdown(
            '<div class="section-note">Predict for a profiled cell line and a screened compound using the saved '
            'CatBoost model. No model training or external API call is performed.</div>',
            unsafe_allow_html=True,
        )
        try:
            with st.spinner("Preparing the saved model and expression features…"):
                features, positions, pca_live, live_model = load_live_artifacts()
            pca_features = [column for column in features if column.startswith("pca_")]
            if not pca_features:
                st.error("No PCA expression features are available in the saved model schema.")
            elif pca_live.empty:
                st.error("The expression feature file contains no cell-line records.")
            else:
                line_ids = sorted(pca_live["ModelID"].dropna().astype(str).unique().tolist())
                if "drug_id" not in ranked.columns:
                    st.error("The ranking CSV has no drug_id column, so a matching live-predictor drug list cannot be built.")
                elif not line_ids:
                    st.error("No valid cell-line ModelID values were found.")
                else:
                    drug_ids = sorted(ranked["drug_id"].dropna().astype(str).unique().tolist())
                    name_of = dict(zip(ranked["drug_id"].astype(str), ranked["drug_name"].astype(str)))
                    default_line = line_ids.index("ACH-000768") if "ACH-000768" in line_ids else 0
                    form_left, form_right = st.columns(2)
                    with form_left:
                        selected_line = st.selectbox("Cell line (ModelID)", line_ids, index=default_line)
                    with form_right:
                        if drug_ids:
                            selected_drug = st.selectbox(
                                "Drug",
                                drug_ids,
                                format_func=lambda drug: f"{name_of.get(drug, drug)} ({drug})",
                            )
                        else:
                            selected_drug = None
                            st.warning("No drug IDs are available in the ranking CSV.")

                    st.caption("The sensitivity threshold below is a project-level heuristic (predicted AUC < 0.30).")
                    if st.button("Predict sensitivity", type="primary", disabled=selected_drug is None):
                        matching_rows = pca_live.loc[pca_live["ModelID"].astype(str) == selected_line]
                        if matching_rows.empty:
                            st.error(f"Expression features were not found for cell line {selected_line}.")
                        else:
                            drug_feature = f"drug_{selected_drug}"
                            if drug_feature not in positions:
                                st.error(
                                    "This drug ID does not have a matching feature in the saved model schema. "
                                    "Check that the ranking and model artifacts come from the same run."
                                )
                            else:
                                import numpy as np

                                feature_vector = np.zeros((1, len(features)), dtype=np.float32)
                                source_row = matching_rows.iloc[0]
                                missing_pca = [column for column in pca_features if column not in source_row.index]
                                if missing_pca:
                                    st.error(f"Missing PCA features in expression data: {', '.join(missing_pca[:5])}")
                                else:
                                    try:
                                        for column in pca_features:
                                            feature_vector[0, positions[column]] = float(source_row[column])
                                        feature_vector[0, positions[drug_feature]] = 1.0
                                        predicted_auc = float(np.clip(live_model.predict(feature_vector)[0], 0, 1))
                                        result_left, result_right = st.columns(2)
                                        result_left.metric("Predicted AUC", f"{predicted_auc:.4f}")
                                        if predicted_auc < 0.3:
                                            result_right.metric("Project-level call", "Sensitive")
                                        else:
                                            result_right.metric("Project-level call", "Resistant / intermediate")
                                    except (TypeError, ValueError, KeyError) as exc:
                                        st.error(f"Could not assemble a valid model input: {exc}")
        except Exception as exc:
            st.warning(
                "The live predictor is unavailable in this deployment. The precomputed ranking, selectivity, "
                "shortlist, combination, and model-audit tabs can still be used."
            )
            st.caption("Expected live artifacts: `tnbc_ml_dataset.parquet`, `expression_pca128_breast.parquet`, and `catboost_tnbc_v1.cbm` in `outputs_local/`.")
            with st.expander("Technical details"):
                st.code(str(exc))

# ---------- footer ----------
st.divider()
st.caption(
    "Reproducibility: DepMap 26Q3 + GDSC2/PRISM-REPURPOSING 27Oct23. "
    "Splits are cell-line holdouts; MDA-MB-231 and MCF7 were not used for tuning. "
    "Consult model_info.json and audit_summary.json for artifact/version details."
)
