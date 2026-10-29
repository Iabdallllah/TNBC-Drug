# FerroSelect / TNBC — Defence Report v1.1

**Dated addendum to the v1.0 master report** · 2026-10-29 · post-hoc, exploratory audit

| | |
|---|---|
| Supersedes | v1.0 master report (delivered 2026-10-28) |
| Does not modify | `outputs_local/sealed_prediction.md`, v1.0 rankings, v1.0 leaderboard, v1.0 artifacts |
| v1.0 reference | git tag `v1.0-defense` (commit `8be8319`) |
| Sealed prediction | dated 2026-10-16 13:53 UTC; file SHA256 `de0922e7…4db851`; internal prediction hash `cc8e2216a19695b1` |
| Evidence tiers | measured (PRISM/CTRPv2/DepMap) · predicted (CatBoost, FerroSelect, RSL3 v2) · hypothesis (8-member panel, combinations, rescue mechanism) |
| Test status | 32/32 passed (`outputs_local/validation_status.json`) |

> **Document status.** This is a dated addendum, not a silent rewrite. The v1.0 master
> report was a briefing; its underlying artifacts are preserved at tag `v1.0-defense`.
> The baseline and selectivity analyses below were designed after the holdout had been
> seen, so they are labelled **post-hoc, exploratory**. The holdout is spent.

---

## 1. Executive summary

We built a leakage-audited drug-response benchmark for TNBC and tested it honestly.
On seven unseen breast lines, a one-line per-drug mean baseline (per-line Spearman
**0.830**, R² **0.825**) outperforms the gradient-boosted CatBoost v1.0 model (per-line Spearman
**0.630**, R² **0.567**). Predicted selectivity is uncorrelated with measured selectivity
(Pearson **0.003**, Spearman **0.027**, n = 1,322), and the restricted check on active
drugs does not rescue it (n = 66, Spearman **0.029**).

We therefore **do not claim a validated cell-line-specific predictor**. The demonstrated
result is narrower and more useful: the current architecture does not beat drug identity.
Within the 227 drugs CatBoost actually learned, it ties the baseline (0.801 vs 0.804)
rather than beating it; that is evidence of no measurable expression lift in this design,
not proof that expression carries no signal.

What survives is the audit machinery that exposed the failure, the external CTRPv2 RSL3
evidence, and a pre-registered, mechanism-driven wet-lab panel. The plate decides the
biology.

---

## 2. What changed in v1.1

| # | Change | Location |
|---|---|---|
| 1 | Per-drug mean baseline placed beside CatBoost; paired line bootstrap; subset and calibration tables | §3, dashboard **Model & audit**, `outputs_local/baseline_audit.*` |
| 2 | Selectivity null reported (all shared, learned 227, restricted AUC < 0.5 either) | §4, dashboard **Selectivity**, `baseline_audit.json` |
| 3 | “Selective” relabelled **“Predicted selective (unvalidated)”** | dashboard, §4 |
| 4 | NDCG@10 definition corrected to graded relevance `1 − AUC`; only Recall@10 is binary `AUC < 0.3` | §5, README |
| 5 | Expression-line count corrected from 77 to **76 unique**; 79 rows / 3 duplicate ModelIDs explained | §6, `audit_summary.json`, `expression_line_audit.json` |
| 6 | `quarantined` column added to the four legacy NAC CSVs; Erastin/Cloperastine regression test | §7, data layer, tests |
| 7 | Four API tests added; suite count now 32/32 and read from `validation_status.json` | §8 |
| 8 | Baseline/selectivity analyses explicitly labelled post-hoc, exploratory | throughout |

No sealed prediction, ranking or v1.0 model artifact was edited.

---

## 3. The baseline result (post-hoc, exploratory)

### 3.1 Method

- Baseline: each drug’s mean AUC on the **23 training lines only**.
- Fallback: drugs with no training row use the global training mean. In this holdout,
  **0 drugs / 0 rows** needed the fallback; all 1,482 drugs have at least one training row.
- Evaluation frame: the **identical 7,336 holdout rows** (7 lines × 1,482 drugs) used for
  CatBoost. The comparison is paired.
- Uncertainty: paired bootstrap resampling the **seven lines**, not the 7,336 rows.
  2,000 resamples, seed 20261029.

### 3.2 Headline metrics

| Model | MAE | RMSE | R² | Pooled Spearman | Per-line Spearman | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CatBoost-800 | 0.0811 | 0.1003 | 0.5669 | 0.4169 | 0.6304 | 0.9411 | 0.6933 |
| **Per-drug mean** | **0.0437** | **0.0638** | **0.8246** | **0.8449** | **0.8295** | **0.9452** | **0.7458** |

### 3.3 Paired bootstrap (CatBoost − baseline)

| Metric | Mean difference | 95% CI | CatBoost better |
|---|---:|---:|---:|
| Per-line Spearman | −0.199 | [−0.250, −0.134] | 0.0% |
| R² | −0.257 | [−0.271, −0.236] | 0.0% |
| NDCG@10 | −0.004 | [−0.014, +0.007] | 23.6% |

The baseline wins per-line Spearman on **all seven lines**, by −0.049 to −0.256.
NDCG@10 is a tie and is a weak discriminator; no claim should rest on it in either
direction.

### 3.4 Subsets and calibration

| Subset | n | CatBoost per-line | Baseline per-line | CatBoost R² | Baseline R² |
|---|---:|---:|---:|---:|---:|
| All holdout | 7,336 | 0.630 | 0.829 | 0.567 | 0.825 |
| Learned 227 drugs | 1,211 | 0.801 | 0.804 | −0.028 | 0.680 |
| Measured AUC < 0.5 | 286 | 0.632 | 0.615 | −3.691 | −0.741 |
| Measured AUC < 0.3 | 55 | 0.430 | 0.455 | −10.844 | −2.967 |

Within the learned 227, it is a **tie**, not a lift. Absolute calibration is also worse:
on those 1,211 rows, measured mean AUC is **0.606**, CatBoost predicts **0.741**, the
baseline predicts **0.633**. The model overpredicts AUC by ~0.135 on the drugs it learned,
which matters for fixed AUC thresholds (0.3, 0.4).

### 3.5 Coverage is not lift

All **55/55** measured AUC < 0.3 holdout rows fall inside the 227 learned drugs, and every
line’s measured top-10 is 10/10 inside that set. Present this as **“the plateau does not
hide the actives”**, not as model strength: the per-drug baseline covers those drugs
trivially.

### 3.6 What this does and does not show

- **Does show:** the CatBoost v1.0 model does not beat a one-line drug-identity baseline; the
  architecture failed as a cell-line-specific predictor.
- **Does not show:** that expression carries no signal. CatBoost collapsed 1,255 of 1,482
  drug identities to a per-line constant, so it never had a fair chance to test expression
  for those drugs.
- The learned-227 tie is the cleanest statement: **no measurable expression lift in this
  design**, not proof of absence.

---

## 4. Selectivity null (post-hoc, exploratory)

Predicted Δ = predicted AUC(MCF7) − predicted AUC(MDA-MB-231); positive means predicted
MDA-selective. Measured Δ uses the same definition on the holdout measurements.

| Analysis | n | Pearson | Spearman | Sign agreement |
|---|---:|---:|---:|---:|
| All shared drugs | 1,322 | 0.0025 | 0.0273 | 52.8% |
| Learned 227 drugs | 214 | −0.0011 | 0.0460 | 49.5% |
| Restricted: measured AUC < 0.5 in either line | 66 | −0.0179 | 0.0287 | 43.9% |

The predicted Δ range is **−0.0557 to +0.0385**, while the MDA-MB-231 80% interval is
**±0.093**. The entire selectivity range is inside the noise band. The restricted check
was requested before the defence; it confirms the null.

### 4.1 The four v1.0 “selective” calls

| Drug | Predicted Δ | Measured Δ | Measured MDA-MB-231 AUC | Measured MCF7 AUC |
|---|---:|---:|---:|---:|
| Triptolide | +0.0339 | −0.0339 | 0.1169 | 0.0831 |
| Dolastatin-10 | +0.0186 | −0.0236 | 0.2071 | 0.1836 |
| Romidepsin | +0.0214 | −0.0002 | 0.2768 | 0.2766 |
| Exatecan-mesylate | +0.0151 | −0.1399 | 0.3412 | 0.2013 |

For Triptolide, Dolastatin-10 and Exatecan-mesylate, MCF7 is measurably **more sensitive**
than MDA-MB-231—the opposite of the predicted sign. Romidepsin is essentially zero
(−0.0002). The badge is relabelled **“Predicted selective (unvalidated)”**. It is a
correct-sign hypothesis to test by SI = IC50(MCF7)/IC50(MDA-MB-231) on the plate, not a
finding.

> Note: an earlier verbal figure for Exatecan’s measured Δ was wrong. The artifact-derived
> holdout pair is −0.1399 (MDA 0.3412, MCF7 0.2013). The conclusion is unchanged: the call
> is unsupported. The report uses the artifact value.

### 4.2 Combination ranks are arithmetic

Bliss gives `E_AB = E_A + E_B − E_A·E_B > max(E_A, E_B)`, so `AUC_combo = 1 − E_AB` is
always below the better component. A combination out-ranking its components is a
mathematical identity, not a discovery. Label these **additivity arithmetic** and require
the 4×4 checkerboard for any synergy claim.

---

## 5. NDCG definition fix

- **NDCG@10:** graded relevance `rel = 1 − AUC`; lower AUC = more relevant. This is what
  `ndcg_at_k` in `TNBC_02_Modeling_Ranking_Colab.ipynb` implements.
- **Recall@10:** binary sensitivity threshold `AUC < 0.3`.

v1.0's metric note was inaccurate: NDCG uses graded relevance `1 − AUC`; only Recall@10
uses the binary `AUC < 0.3` threshold. The correction does not change any ranking
or metric value, but it matters for interpreting the score: NDCG near 0.94 is saturated
for both models and its paired CI spans zero.

---

## 6. Expression-line and PCA accounting (v1.1 correction)

| Quantity | v1.1 value | Explanation |
|---|---:|---|
| Breast lines annotated | 96 | DepMap 26Q3 |
| Unique breast lines with expression | **76** | v1.0 said 77; that was a row/counting artifact |
| Rows in shipped PCA table | 79 | 3 ModelIDs duplicated: MCF7, ACH-001396, ACH-001662 |
| PCA fit rows | 62 | |
| PCA fit unique lines | 60 | 60 + 16 = 76 |
| PCA holdout unique lines | 16 | includes all 7 ML holdout lines |
| ML holdout lines inside PCA fit | 0 | no leakage |

The phantom-header bug shifted the PCA holdout draw, but it did **not** leak the seven model
holdout lines into the PCA fit. PCA is unsupervised, so any leak would have been mild; this
check closes the question.

---

## 7. Provenance quarantine and collision hardening

| Panel drug | Matched L1000 term | Signature genes | Status |
|---|---|---:|---|
| Erastin | Erastin Down | 244 | Valid |
| Sulfasalazine | Sulfasalazine Down | 244 | Valid |
| Deferoxamine | Deferoxamine-Mesylate Down | 245 | Valid |
| Simvastatin | Simvastatin Down | 244 | Valid |
| Lovastatin | Lovastatin Down | 247 | Valid |
| N-acetylcysteine | **Solifenacin-Succinate Down** | 245 | **QUARANTINED** |
| RSL3 | none | — | Absent from L1000 |

- The legacy matcher resolved the abbreviation “NAC” inside “Solife**NAC**in”.
- The four CSVs that carry legacy NAC rows now have a **`quarantined`** column:
  `offtarget_panel_map.csv`, `offtarget_rescue_reversal.csv`,
  `protein_array_shortlist.csv`, `offtarget_coverage_ferroselect.csv`.
- The UI withholds the NAC rescue arm; the tests assert the marker.
- Collision sweep over 5,425 L1000 base terms: `nac` matches 20+ terms; `erastin` is a
  latent substring of `Cloperastine`. Exact match wins today, and the resolver now requires
  a token-boundary match for any fallback. A regression test asserts Erastin resolves to
  Erastin (or to nothing), never to Cloperastine.

DFO is the validated rescue arm. The wet-lab NAC molecule is unaffected; only the
mislabelled in-silico signature is withheld.

---

## 8. Validation status and API

- Full suite: **32/32 passed**, 15 warnings, ~15 s.
- `outputs_local/validation_status.json` is written by `tests/conftest.py`; the report
  reads the count instead of typing it.
- Four API tests now cover `/health`, `/model-info`, `/predict` parity with the sealed
  ranking, and `/rank` length/order. `/rank-pairs` is covered by the fourth test.
- Manual smoke test on port 8010: `/health` ok; `/model-info` 1,544 features / 76 lines /
  1,482 drugs; `/predict` MDA-MB-231 × Triptolide = 0.27643; `/rank` top 3 =
  Triptolide, Dolastatin-10, Romidepsin; `/rank-pairs` returns additivity pairs.
- The API still takes a pre-profiled `cell_line_id`, not an uploaded expression profile.
  This is a deliberate scope reduction from PRD FR-07/08/10, not a completed upload path.

---

## 9. Revised defence Q&A

**Q1. “Your CatBoost model loses to a column mean. Why is it in your project?”**

> It is reported as a negative result. The baseline wins because CatBoost never learned
> 85% of drug identities. Within the 227 drugs it did learn, it ties the baseline. The
> expression contribution is unproven, and we say so. The project’s value is the audit
> that found this, not the model.

**Q2. “Did your sealed prediction or your ‘selective’ labels rely on this model?”**

> The panel was chosen on mechanism (GPX4, system xc⁻, iron) and agreed with the lab
> before any plate. The model outputs are a frozen, falsifiable record, not the
> justification. Predicted selectivity is uncorrelated with measured selectivity, so the
> four “selective” drugs are relabelled unvalidated hypotheses. Selectivity will be tested
> by SI = IC50(MCF7)/IC50(MDA-MB-231) on the plate. We will report Spearman with a
> bootstrap CI after unblinding, as the PRD requires.

**Q3. “If the model is this weak, what is the scientific contribution?”**

> A reproducible pipeline whose own controls caught its failures: a circular-label
> inflation of 0.65–0.99, a substring bug that would have mislabelled Solifenacin as NAC,
> an 84.7% plateau, and a dead selectivity threshold. We found them, bounded them and
> tested the fixes. We also pre-registered a falsifiable wet-lab panel, with CTRPv2
> showing MDA-MB-231 in the most RSL3-sensitive 30% while MDA-MB-468 is resistant. A
> reviewer can trust this report’s numbers more than a clean-looking one, because the
> failures are documented.

**Q4. “Is 55/55 sensitive-drug coverage a model strength?”**

> No. It is coverage: the plateau does not hide the measured actives. The per-drug
> baseline covers them trivially. It is a useful sanity check, not evidence of lift.

**Q5. “Why should I trust the DFO rescue arm if the NAC matcher failed?”**

> Because the provenance table shows the requested agent, the matched L1000 term and the
> status for every panel member. DFO resolves to Deferoxamine-Mesylate, not to a
> substring. NAC resolves to Solifenacin and is quarantined. The failure is contained,
> marked in the data layer, and regression-tested.

---

## 10. Slide arc

1. Problem: TNBC heterogeneity; the selectivity hypothesis.
2. Data and leakage controls: whole-line holdout, GroupKFold, PCA fit/holdout audit.
3. Models: Dummy → XGBoost → CatBoost; gates.
4. **Baseline comparison up front:** CatBoost 0.630 vs per-drug mean 0.830; R² 0.567 vs
   0.825; paired CI; NDCG tie.
5. Audit findings: plateau (1,255/1,482), NAC quarantine, Aim-0 circularity, selectivity
   null.
6. External evidence: CTRPv2 RSL3; MDA-MB-231 top 30%, MDA-MB-468 resistant; RF 0.462
   vs 0.83 ceiling.
7. The panel as a pre-registered hypothesis: mechanism-selected, model outputs frozen.
8. Limitations and next steps: drug descriptors, drug × cell-line interactions,
   multi-task models, more lines, nested selection.

---

## 11. What survives and what is withdrawn

**Survives**

- Leakage-audited pipeline: whole-line holdout, GroupKFold, no PCA leakage.
- Seven-tab dashboard; 32/32 tests; zero AppTest exceptions in tested paths.
- NAC quarantine, plateau disclosure, Aim-0 circularity check.
- External CTRPv2 RSL3 evidence: MDA-MB-231 top 30%, MDA-MB-468 resistant.
- 100% coverage of measured AUC < 0.3 drugs by the learned 227 (coverage only).
- API parity and the zero-JS defence report.
- The mechanism-driven, pre-registered panel as a hypothesis.

**Withdrawn or relabelled**

- “CatBoost is a cell-line-specific predictor.” Withdrawn: it is worse than a per-drug mean.
- “R² 0.567” as a headline without the 0.825 baseline.
- “NDCG 0.941” as a win; the baseline is 0.945 and the difference is not significant.
- “The four drugs are selective.” The measured deltas are opposite or zero.
- “The panel ranking is model-supported.” It is a mechanism hypothesis; combos outrank by
  arithmetic.
- “Expression features add value.” Not demonstrated; the current model does not beat drug
  identity.
- “Blind selection.” Withdrawn: the v1.0 model was selected on the same holdout; disclose mild optimism.
- “Percentage of cells killed.” Withdrawn: outputs are kill proxies; no MTT results exist.

---

## 12. Sealed record and access control

The sealed record is the supervisor’s. Before plate 1:

- [ ] Print and sign `outputs_local/sealed_prediction.md`; lodge the signed copy with the
      supervisor.
- [ ] Record the timestamp and hash in `outputs_local/sealed_prediction_manifest.json`.
- [ ] Confirm who had dashboard access before plate 1. The bench team must not see the
      **Defense report** or **Live predictor** tabs until unblinding.
- [ ] Freeze the git tag for the defence release; do not retrain after the seal.
- [ ] The supervisor signs the negative-result framing; the computational lead owns the
      numbers.

---

## 13. Limitations and next steps

- The holdout is spent; any new model scored on it is contaminated. New models must use
  leave-line-out CV within the 23 training lines, or be labelled exploratory.
- Effective n is 7 lines, not 7,336 rows. Row-pooled CIs are optimistic; the paired line
  bootstrap is the robust comparison.
- The learned-227 tie is not proof of absence. A drug-descriptor × expression interaction
  model is the right next test, not another CatBoost on the same features.
- Optional time-boxed diagnostic (pre-declared): drug mean + per-line offset predicted
  from PCA, and drug mean + kNN over lines in PCA space. Success criterion: beats the
  per-drug mean on per-line Spearman in leave-line-out CV on the 23 training lines. A null
  result strengthens the audit framing; it is not required for the defence.
- The upload/QC/user-guide FR-07/08/10 path remains deliberately out of scope for this
  defence. Say so explicitly.

---

### Artifact map

| Artifact | Purpose |
|---|---|
| `outputs_local/baseline_audit.json` | Full baseline, bootstrap, subset, calibration, coverage, selectivity, limitation record |
| `outputs_local/baseline_audit.csv` | CatBoost vs per-drug mean table used by the dashboard/report |
| `outputs_local/expression_line_audit.json` | 76/79/62/60/16 reconciliation and PCA holdout proof |
| `outputs_local/validation_status.json` | Test count/status generated by `tests/conftest.py` |
| `outputs_local/defense_report.html` | v1.1 zero-JS printable report |
| `src/models/baseline_audit.py` | Reproducible post-hoc audit script |
| `tests/test_pipeline.py` | 32-test suite, including four API tests and the quarantine/Erastin checks |

*Prepared for the defence. The plate decides the biology.*

### 7. Diagnostic (post-hoc, exploratory)
Pre-registration 57b363e → results 8e903c4 (23 train lines, holdout excluded, label everywhere: post-hoc, exploratory). Honest null: M1=M0; M2 no evidence (CI spans zero); M3 skipped.

