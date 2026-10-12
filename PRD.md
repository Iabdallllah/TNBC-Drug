# PRD — TNBC Drug Response Predictor

**Version:** 1.0 (MVP for Graduation)
**Date:** 2026-09-23
**Status:** Draft
**Scope lock:** GDSC ranking only / Laptop-Colab / <6 months

---

## 1. One-liner

> An AI system that predicts and ranks GDSC drug sensitivity for Triple-Negative Breast Cancer models from cell-line molecular profiles, validated experimentally on MDA-MB-231 (TNBC) vs MCF7 (ER+ control).

**Not** a patient-treatment recommender. **Not** a novel-drug generator.

---

## 2. Problem

Triple-Negative Breast Cancer (TNBC) lacks ER / PR / HER2 targets, so response varies widely across models. Question:

1. Can a molecular profile (expression + selected mutations) predict drug sensitivity (AUC) in breast cell models?
2. Do top AI-ranked drugs show stronger killing on MDA-MB-231 than MCF7 in vitro (selectivity)?

MCF7 is intentionally ER+ / Luminal — used as non-TNBC control, not as TNBC.

---

## 3. Goals / Non-Goals

### Goals (must-have for graduation)
1. Build TNBC-aware ML dataset from public data (GDSC + DepMap).
2. Train regression model: `molecular + drug -> AUC`.
3. Rank GDSC drugs for MDA-MB-231.
4. Explain predictions with SHAP.
5. Serve via FastAPI + Streamlit dashboard.
6. Wet-lab validate Top 3-5 candidates on MDA-MB-231 vs MCF7.
7. Report Predicted vs Experimental correlation.

### Non-Goals (explicitly out of MVP)
- Prediction for unseen / novel chemical structures (no drug fingerprints in V1).
- Patient-level / clinical prediction.
- Multi-omics (proteomics, methylation) — V3 only.
- React frontend, Postgres, Docker prod deploy — overkill for MVP.
- Deep Learning as primary model (laptop constraint).

---

## 4. Inputs / Outputs

### 5.1 Input (MVP)
```json
{
  "cell_line_id": "MDA-MB-231",
  "expression": { "TPM vector or precomputed PCA vector": "..." }
}
```
V1 uses **expression only**. Mutations added in V2 if time allows.

Source of truth for MDA-MB-231 / MCF7 expression: DepMap `OmicsExpression` (TPM, log2).

### 5.2 Output
```json
{
  "cell_line": "MDA-MB-231",
  "model_version": "catboost-v1.2",
  "ranking": [
    {"rank": 1, "drug": "Drug A", "predicted_AUC": 0.18, "ci_lower": 0.15, "ci_upper": 0.22},
    {"rank": 2, "drug": "Drug B", "predicted_AUC": 0.27}
  ],
  "explanation": {"top_genes": ["gene X", "gene Y"], "shap_plot": "url"}
}
```
Lower AUC = more sensitive. Never display "this drug will cure cancer". Display: `predicted sensitivity in cell model`.

---

## 5. Functional Requirements

### FR-01 Data Ingestion
- Load GDSC2 fitted drug response (`AUC`, `LN_IC50`).
- Load DepMap `sample_info`, `OmicsExpression`, `OmicsMutations` (V2).
- Log dataset release versions (e.g. DepMap 24Q4, GDSC2 v8.x).

### FR-02 Data Cleaning
- Deduplicate cell_line x drug pairs.
- Unify IDs via Cell Model Passport / COSMIC ID, not name string.
- Handle missing expression (drop genes with >20% missing, median impute rest).
- Log data-quality issues to `data_quality_report.md`.

### FR-03 TNBC Annotation
- Do NOT filter by name only. Use `sample_info` + literature list for TNBC label.
- Keep columns: `cell_line_id, name, cancer_type=breast, subtype=[TNBC, Luminal, HER2+, other]`.
- Must contain MDA-MB-231 (TNBC) and MCF7 (Luminal control).

### FR-04 Dataset Integration
Join key: `cell_line_id + drug_id` -> one row per pair.
```text
ML_ROW = [pca_0..pca_127, drug_one_hot, auc_target, ln_ic50_secondary]
```

### FR-05 Training
Support in order:
1. `DummyMean` baseline (must beat this)
2. RandomForest
3. XGBoost
4. CatBoost (expected best for MVP)

CV: `GroupKFold by cell_line_id, k=5`. Holdout: MDA-MB-231 + MCF7 + 20% TNBC lines fully unseen.

### FR-06 Evaluation
Regression: MAE, RMSE, R², Spearman.
Ranking (primary for project): Spearman per-cell-line, NDCG@10, Recall@10 for sensitive drugs (AUC < 0.3).
Requirement: report metrics both globally AND on TNBC-only slice.

### FR-07 Explainability
SHAP global (top 20 genes) + local (per MDA-MB-231 prediction). Save plots to `experiments/`.

### FR-08 Prediction API
```
POST /predict  -> single drug AUC
POST /rank     -> ranked list for cell_line, top_k param
GET  /health
GET  /model-info -> {model_version, dataset_version, metrics}
```

### FR-09 Experimental Logging
Store:
```text
drug, predicted_AUC, tested_concentrations, experimental_IC50, experimental_AUC, viability_csv_path
```
Comparison figure: `predicted vs experimental` scatter + Spearman.

---

## 6. Non-Functional Requirements

- **Reproducibility:** every run logs `dataset_version + code_commit + params + seed` in MLflow.
- **Traceability:** prediction response includes `model_version`.
- **Performance:** train on Colab CPU <30 min, inference <500ms.
- **Privacy:** no patient data, no PII.
- **Portability:** `requirements.txt` + `parquet` files, no mandatory DB / Docker for MVP.

---

## 7. Data Architecture

```text
data/raw/gdsc/      # GDSC2 fitted CSV
data/raw/depmap/    # expression, sample_info
  ↓
data/silver/        # cleaned parquet
  ↓
data/gold/tnbc_ml_dataset.parquet
```

Validation with Pandera (light): check `0 <= AUC <= 1`, no NaN targets.

---

## 8. ML Architecture V1 (locked)

```text
Expression (19k genes)
  -> variance filter top 2000
  -> log2(TPM+1) + z-score per gene
  -> PCA 128 (fit on train cell lines only)
  + Drug one-hot (GDSC ~297 drugs)
  ↓
XGBoost / CatBoost regressor -> AUC
```

Why AUC first: bounded [0,1], available in GDSC fitted data, comparable across drugs. `LN_IC50` as secondary model only if time.

Why no deep learning: n_samples (~10k breast pairs, ~200k pan-cancer) << n_features, laptop, <6mo.

Future V2: + selected mutations (TP53, BRCA1/2, PIK3CA, PTEN), + CNV, + PRISM external check.

---

## 9. MVP Definition of Done

- [ ] `01_data_audit.ipynb` shows intersection counts + MDA-MB-231/MCF7 present
- [ ] `tnbc_ml_dataset.parquet` built reproducibly
- [ ] CatBoost beats DummyMean on unseen cell lines (R²>0.25 and per-line Spearman>0.4 on breast slice — threshold to confirm after audit)
- [ ] Ranked Top-20 list for MDA-MB-231 exported
- [ ] SHAP plot + FastAPI + Streamlit working on Colab
- [ ] Wet-lab: dose-response for Top 3-5 + Doxorubicin (pos control) on both lines
- [ ] Final figure: predicted vs experimental

---

## 10. Wet-Lab Validation Protocol (summary)

1. Candidates: Top 3-5 AI + 1 positive control (Doxorubicin/Paclitaxel) + 1 negative (predicted resistant).
2. Assay: viability (MTT / CellTiter-Glo), 72h, 6-8 concentrations, triplicates.
3. Cell lines: MDA-MB-231 and MCF7 side-by-side, same passage window.
4. Readout: IC50 + AUC from fitted curve.
5. Success: Spearman(predicted_AUC, experimental_AUC) > 0 and selectivity `IC50_MCF7 / IC50_MDA231 > 1.5` for at least 1 candidate. This is research signal, not clinical claim.

---

## 11. Risks

| Risk | Mitigation |
|------|------------|
| Only ~20 TNBC lines -> overfit | Train pan-cancer/breast, test TNBC slice; PCA + regularization |
| MDA-MB-231 missing in GDSC/DepMap intersection | Fallback: use CCLE expression + map by name + Passport ID; document version |
| Drug one-hot can't generalize to new drugs | Accepted; scope locked to GDSC ranking only |
| Batch effect GDSC vs lab | Use rank order, not absolute AUC; include controls |
| Colab RAM limits (expression matrix) | Use parquet + float32 + PCA, chunked load |

---

## 12. Milestones (20 weeks)

- W1-2: Data audit
- W3-5: Cleaning + gold dataset
- W6-8: Baseline + XGBoost/CatBoost + holdout eval
- W9-10: SHAP + API + Streamlit
- W11-18: Wet-lab (parallel with writing)
- W19-20: Thesis + defense figures

---

## 13. Open Questions

1. DepMap release version to freeze? (propose latest 2024+)
2. Exact GDSC2 file version?
3. Lab max concentrations / available drugs budget?
4. Thesis language: EN or AR?

---

## Appendix: Repo Structure

```text
TNBC/
├── PRD.md
├── data/raw/ silver/ gold/
├── notebooks/01_data_audit.ipynb
├── src/ingestion/ features/ models/ api/
├── experiments/
├── dashboard/app.py
└── wetlab/results.csv
```
