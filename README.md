# TNBC Drug Response Predictor

An end-to-end computational platform that predicts and ranks drug sensitivity
for **Triple-Negative Breast Cancer (TNBC)**, validated experimentally for
selectivity against a non-TNBC control.

**One-liner:** given a breast cancer cell line's molecular profile, the system
predicts its sensitivity (AUC) to every screened drug, ranks the strongest
killers of the TNBC model **MDA-MB-231**, and checks selectivity against the
luminal control **MCF7** — first computationally, then in the wet lab.

> Outputs are *predicted sensitivities in cell models*, never clinical
> treatment recommendations.

---

## 1. Biological premise (for the bio team)

TNBC lacks the three receptors (ER, PR, HER2) targeted in other breast cancers,
so drug response varies widely between models. The project tests one hypothesis:

> **Top computationally ranked drugs kill MDA-MB-231 (TNBC) preferentially
> over MCF7 (ER+, luminal), quantified by the Selectivity Index
> SI = IC50(MCF7) / IC50(MDA-MB-231).**

The ML stream owns everything computational. The bio stream owns receptor
annotation, drug mechanism review, and target/pathway confirmation. The wet-lab
stream owns cell culture, MTT viability assays, and dose-response fitting.

---

## 2. Repository structure

```text
TNBC-Drug/
├── TNBC_01_Data_Audit_Colab.ipynb      # data audit + gold dataset (run first)
├── TNBC_02_Modeling_Ranking_Colab.ipynb # training + ranking + SHAP (run second)
├── src/
│   ├── ingestion/download.py            # fetch GDSC2 + DepMap raw files
│   ├── models/combo_rank.py             # Bliss 2-drug combo ranking (CLI)
│   └── api/main.py                      # FastAPI: /predict /rank /rank-pairs
├── dashboard/app.py                     # Streamlit UI (run: streamlit run dashboard/app.py)
├── tests/test_pipeline.py               # leakage + schema + artifact + gate tests
├── outputs_local/                       # all v1 results (rankings, models, SHAP, leaderboard)
├── Data/                                # local raw datasets (git-ignored, see below)
├── requirements.txt                     # frozen working environment
├── PRD.md                               # product requirements (frozen v1.0 scope)
└── project_blueprint.html               # 20-week team execution blueprint
```

---

## 3. Quickstart

### Option A — Google Colab (recommended for the team)

1. Upload the small files (`Model.csv`, `REPURPOSING*.csv`) via the Files panel.
2. Paste a **fresh** DepMap signed URL into the Section 2 download cell to fetch
   `OmicsExpressionTPMLogp1HumanAllGenes.csv` (~938 MB, 26Q3). Signed URLs
   expire — generate one right before downloading.
3. Run notebook `01` top to bottom, download its 5 outputs
   (`tnbc_ml_dataset.parquet`, `holdout_lines.npy`,
   `breast_lines_annotated.csv`, `expression_pca128_breast.parquet`,
   `audit_summary.json`).
4. Open notebook `02` in a new session, upload those 5 files plus
   `REPURPOSINGLog2ViabilityCollapsedConditions.csv`, run top to bottom.

### Option B — Local

```bash
pip install -r requirements.txt
python src/ingestion/download.py --small-only   # GDSC2 + Model.csv (~22 MB)
python src/ingestion/download.py --all          # + 1.1 GB expression matrix
streamlit run dashboard/app.py                  # UI on http://127.0.0.1:8501
uvicorn src.api.main:app --host 127.0.0.1 --port 8000   # API
python -m pytest tests/ -q                      # 7 integrity tests
```

---

## 4. Data sources (all public, academic-free)

| Source | File | Content | Version used |
|---|---|---|---|
| DepMap (Broad) | `OmicsExpressionTPMLogp1HumanAllGenes.csv` | log-TPM expression, ~60k genes | Public 26Q3 |
| DepMap (Broad) | `Model.csv` | cell-line metadata (lineage, IDs) | Public 26Q3 |
| GDSC (Sanger) | `GDSC2_fitted_dose_response_27Oct23.xlsx` | fitted AUC + LN_IC50, 969 lines x 286 drugs | release 8.5 |
| PRISM (Broad) | `REPURPOSING*Matrix.csv` | repurposing viability screens incl. non-oncology drugs | 19Q4/20Q2 |

Raw files live in `Data/` and are **git-ignored** (too large for GitHub).
`src/ingestion/download.py` reproduces them from the sources above.

---

## 5. How it works (pipeline)

```text
raw CSVs → ID-unified cleaning → variance filter + PCA-128 (fit on train folds only)
        → gold parquet (cell_line x drug → AUC)
        → Dummy / XGBoost / CatBoost with GroupKFold by cell line
        → TreeSHAP → ranked killers + Bliss combos → API + dashboard
```

**Non-negotiable guardrails** (enforced in code and in `tests/`):

- Splits are **always by cell line** (`GroupKFold` on `cell_line_id`) — never random rows.
- **MDA-MB-231 and MCF7 are blind holdouts**: excluded from all tuning and from
  every PCA/scaler fit.
- Feature Store discipline: PCA/scaler/median statistics are fit on
  non-holdout rows only, then applied to the rest.
- Every prediction carries a `model_version`; every run is reproducible from a
  pinned `requirements.txt` with fixed seeds.

---

## 6. v1 results snapshot

| Metric (unseen cell lines) | CatBoost-800 | Gate |
|---|---|---|
| R2 | 0.567 | > 0.25 |
| Per-line Spearman | 0.630 | > 0.4 |
| NDCG@10 | 0.941 | ranking primary |

- Training set: **30,994** unique (cell line, drug) pairs — 30 lines x 1,482 drugs.
- Top predicted killers of MDA-MB-231: **Triptolide, Dolastatin-10, Romidepsin,
  Echinomycin, Exatecan** (see `outputs_local/mda_mb_231_top20_killers.csv`).
  Selectivity deltas vs MCF7 are modest (~0.02), reported as-is.
- Top Bliss pair: **Triptolide + Maytansinol-isobutyrate** (gap 0.025).
  Additivity assumption — synergy verdict requires checkerboard validation.
- Full artifacts: `outputs_local/` (leaderboard, rankings, models, SHAP plots,
  `model_info.json`, `audit_summary.json`).
- PCA gene loadings: `pca_loadings.parquet` + `pca_loadings_top.csv` map every
  `pca_N` feature to its top-3 loading genes (`pca_0 (ETS1, SPDEF, MSN)` ...).
  Loadings are reproduced by `src/models/pca_loadings.py` and **verified per
  component against the shipped PCA scores** (61/62 components, reconstruction
  error <= 1e-3). `pca_61` is the zero-variance residual direction of the sample
  space — carries no signal, left unattributed and flagged in the dashboard.
  Provenance: the original run wrote a stray header into `wanted_breast_ids.csv`,
  which shifted the holdout draw (see `pca_loadings_meta.json`); the notebook bug
  is fixed, and reproducing the shipped basis still requires the historical draw
  (`--pool phantom`, the default).

---

## 6B. FerroSelect track (proposal-aligned)

The general platform above stays intact. This track implements the funded
proposal's computational half verbatim: same data, proposal-mandated methods.

- **Inducer panel audit** (`ferroptosis_sample_audit.csv`): per-agent training
  rows stated honestly — sulfasalazine 9, simvastatin 22, lovastatin 19,
  erastin 14. RSL3/ML162/ML210 are absent from our matrices (CTRP download
  pending; RSL3 is droppable per the proposal).
- **Proposal-faithful models** (`src/models/ferroselect.py`): Lasso, Ridge and
  RandomForest with nested cell-line GroupKFold, screen covariate slot, and an
  automatic ceiling check against published GPX4-inhibitor agreement
  (max 0.83 — anything above is flagged as a bug, not a triumph).
- **Uncertainty-first outputs** (`ferroselect_panel_predictions.csv`): every
  prediction for the 3 proposal lines carries its training-row count; anything
  under 25 rows is flagged LOW (lab data decides).
- **Network pharmacology** (`src/network/build_network.py`): curated
  drug-target-pathway graph (31 nodes). Both drug routes provably converge on
  GPX4 and lipid hydroperoxides (`network_paths.json`); figure + supplement
  table in `outputs_local/`.
- **Sealed prediction** (`outputs_local/sealed_prediction.md`): frozen-model
  ranking for the 3 lines + nominated statin pairing + SHA256 + sign-off block.
  Print, sign and lodge it BEFORE the first plate — any later training change
  voids the seal.

Pending (blocked on downloads, not on code): raw expression matrix for the
small FerrDb-gene model + Aim-0 ssGSEA baseline, CTRP bulk for RSL3 coverage.

### 6C. Off-target protein-effect layer (additive, computed, wet-lab-validated)

The panel kills the cell; this layer predicts what ELSE each panel member does
to the proteins of MDA-MB-231 — the side-effect profile — and proves the rescue
logic in silico before the plates are poured.

- **Coverage audit** (`offtarget_coverage_ferroselect.csv`): 6/7 panel members
  have LINCS L1000 chem-pert consensus signatures (Erastin, Sulfasalazine,
  N-acetylcysteine, Deferoxamine, Simvastatin, Lovastatin). RSL3 is absent from
  L1000 (66k terms), ChEMBL and GEO (no published RSL3-on-MDA-MB-231 data).
- **RSL3 recovered** (`rsl3_signature.py` -> `rsl3_mini_signature.csv`): from
  GSE247883 (A549, DMSO / RSL3 / RSL3+Fer-1, 24h, n=3 each) — top-150 Up/Down
  signature, labelled single-paper. 38 RSL3-suppressed proteins are expressed in
  MDA-MB-231 (ECM/core: IGFBP4, SPARC, COL1A1/2, COL6A1/2, MMP14, FN1).
  Fer-1 (the in-dataset rescue arm) does NOT reverse the transcriptional
  suppression (12% recovery, MW p=1.0) — it blocks death, not the 24h
  transcriptional state. Reported as-is.
- **Suppression map** (`offtarget_panel_map.csv`): per drug x line, the Down/Up
  signature genes intersected with actually-expressed genes (log2TPM+1 > 1).
  Every panel member suppresses 154–206 expressed proteins in MDA-MB-231.
- **In-silico rescue** (`offtarget_rescue_reversal.csv`): every killer x rescue
  pair reverses significantly (e.g. Sulfasalazine x NAC: 19 genes, 9.2x
  enrichment; RSL3 x NAC: 13.6x, p=7.3e-5). Deferoxamine does not reverse RSL3
  (p=0.41) — consistent with mechanism (GSH restores GPX4's substrate; iron
  chelation acts upstream). GPX4 itself is expressed in MDA-MB-231 (8.24
  log2TPM+1) and ACSL4 is 5.77 vs 0.31 in MCF7 — the ferroptosis-competence gap
  that justifies the MDA-MB-231 focus.
- **Protein-array shortlist** (`protein_array_shortlist.py` ->
  `protein_array_shortlist.csv`): 1,569 candidates across the 7 panel members
  (single agents + killer x statin additive unions), ranked rescue-competent
  first, then membrane/secreted (UniProt-annotated), then effect size.

### 6D. CTRPv2 external ground truth for RSL3 (v2 track; sealed v1 untouched)

PRISM never profiled RSL3, so the sealed v1 ranking cannot score it. CTRPv2.1
(Nat Chem Biol 2016, NCI CTD² portal) DID: the published AUC table covers
RSL-3 (609060) across 808 lines plus ML210/ML162/erastin/CIL56/simvastatin/
lovastatin (sulfasalazine is absent from CTRP). `src/models/ctpv2_rsl3.py`
ingests those measurements, maps 831/835 CTRP line names → ACH ids via
`Data/Model.csv`, and projects every line into the SAME pca_0..61 feature
space v1 consumed (the basis is re-derived in fp64 and verified both against
the published loadings — min |r| > 0.999 on all 61 attributable components —
and against the shipped scores on pca_0..pca_58 of the refreshed file;
near-zero-variance comps 59/60/61 are kept but flagged).

Artifacts (outputs_local/, v2 namespaces only): `ctpv2_ferroptosis_auc.csv`
(5,078 measured AUC rows, `auc_raw` CTRP units + `auc_norm_0_1 = raw/20`,
lower = more cell death), `ctpv2_panel_train.parquet` (4,959 lines × 7 drugs
with 62 pca features), `ctpv2_meta.json`, `rsl3_v2_report.json` and
`rsl3_v2_gold_predictions.csv` (GroupKFold-5-by-line OOF for the proposal
trio; RandomForest wins at OOF Spearman 0.462, 95% CI 0.400-0.518 —
far below the published lab-vs-CTRP RSL3 agreement ceiling 0.83), plus the
fitted artifact `rsl3_v2_winner.joblib` — load with `joblib.load` and call
`model.predict(pca_matrix)` where `pca_matrix` is the 62-column
pca_0..pca_61 projection of the line's expression (same recipe as v1).

Gold lines, measured (raw CTRP AUC → 0-1 scale → % of cohort less sensitive):
MDA-MB-231 6.18 → 0.31 → 30% most-sensitive third (mesenchymal/GPX4-dependent,
matching Viswanathan 2017); MDA-MB-468 10.32 → 0.52; MCF7 11.06 → 0.55.
The FerroSelect view shows this block under "CTRPv2 external ground truth
for RSL3"; nothing sealed is modified.

---

## 7. API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

| Endpoint | Input | Output |
|---|---|---|
| `GET /health`, `GET /model-info` | — | version, feature/drug counts |
| `POST /predict` | `cell_line_id`, `drug_id` | `predicted_auc` |
| `POST /rank` | `cell_line_id`, `top_k` | ranked drugs with sensitivity calls |
| `POST /rank-pairs` | `top_singles`, `top_pairs` | Bliss pairs with selectivity gaps |

Single-drug predictions from notebook 02 and from the API are bit-identical
(verified: Triptolide 0.2764, Dolastatin-10 0.3488, Romidepsin 0.3594).

---

## 8. Wet-lab handoff (what the lab receives / returns)

**Lab receives** (`outputs_local/`):

- `wetlab_shortlist.csv` — Top-5 AI candidates + Doxorubicin/Paclitaxel
  positive controls + 1 predicted-resistant compound.
- `combo_rankings.csv` — Top-5 Bliss pairs (max **2 pairs** to test).

**Protocol:** MTT/CellTiter-Glo, 72h, 6-8 concentrations, triplicates, both lines
side by side. Combos as **4x4 checkerboard only** (16 wells x 3 replicates = 48;
one 96-well plate fits 2 pairs + controls).

**Lab returns:** `raw_absorbance.csv` (drug, cell line, concentration, viability,
replicate) → fitted to 4PL curves for experimental IC50/AUC, then correlated
against predictions (Spearman) with per-drug Selectivity Indexes.

---

## 9. Status by stream

| Stream | Status |
|---|---|
| ML / Data / API / UI | **Done**: pipeline, models, rankings, combos, API, tests (7/7), dashboard |
| Bioinformatics | Pending: confirm distinct targets/pathways for shortlisted pairs |
| Wet-lab | Pending: MTT results (`raw_absorbance.csv`) for Top candidates + 2 pairs |

Success criteria: computational (beat DummyMean on unseen lines — **met**),
experimental (predicted-vs-actual Spearman > 0 and SI > 1.5 for at least one
candidate — **awaiting lab data**).

---

## 10. Reproducibility

- Dataset versions: DepMap Public 26Q3, GDSC2 fitted 27Oct23 (release 8.5).
- Seeds fixed (42) across splits, PCA, and model training.
- `tests/test_pipeline.py` re-verifies leakage guards, schema, artifacts, and
  PRD gates on every run.
