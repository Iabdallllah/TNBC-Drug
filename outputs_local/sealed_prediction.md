# Sealed prediction — FerroSelect computational half

Date (UTC): 2026-10-16 13:53  |  Model: RandomForest (FerroSelect track, nested cell-line CV)  |  Seed: 42
Dataset: DepMap 26Q3 + PRISM-REPURPOSING (n=64 panel rows, 22 lines)  |  Predictions SHA256: `cc8e2216a19695b1`

## Predicted sensitivity (AUC; lower = more sensitive)

| Line | Agent | Predicted AUC | Training rows | Confidence |
|---|---|---|---|---|
| MDA-MB-231 | SULFASALAZINE | 0.8874 | 9 | LOW (<25 rows — lab data decides) |
| MDA-MB-231 | SIMVASTATIN | 0.8844 | 22 | LOW (<25 rows — lab data decides) |
| MDA-MB-231 | LOVASTATIN | 0.8870 | 19 | LOW (<25 rows — lab data decides) |
| MDA-MB-231 | ERASTIN | 0.8644 | 14 | LOW (<25 rows — lab data decides) |
| MDA-MB-468 | SULFASALAZINE | 0.9306 | 9 | LOW (<25 rows — lab data decides) |
| MDA-MB-468 | SIMVASTATIN | 0.9200 | 22 | LOW (<25 rows — lab data decides) |
| MDA-MB-468 | LOVASTATIN | 0.9370 | 19 | LOW (<25 rows — lab data decides) |
| MDA-MB-468 | ERASTIN | 0.8778 | 14 | LOW (<25 rows — lab data decides) |
| MCF7 | SULFASALAZINE | 0.9353 | 9 | LOW (<25 rows — lab data decides) |
| MCF7 | SIMVASTATIN | 0.9250 | 22 | LOW (<25 rows — lab data decides) |
| MCF7 | LOVASTATIN | 0.9418 | 19 | LOW (<25 rows — lab data decides) |
| MCF7 | ERASTIN | 0.9177 | 14 | LOW (<25 rows — lab data decides) |

Frozen-model pooled OOF Spearman: 0.1947 (ceiling check: ceiling check OK (below 0.83 max published agreement))

## Nominated statin pairing

**Simvastatin (fixed sub-lethal dose, ~IC20 set in pilot) + sulfasalazine dose-response**, on the least sensitive line per the table above. Rationale: simvastatin has the most training rows (22) of the statins; sulfasalazine is the cheap Type-I workhorse. This nominates the pairing to TEST — it does not claim predicted synergy.

## Sign-off (human step — print, sign, lodge BEFORE first plate is seeded)

- Computational pair: ______________________  Date: __________
- Supervisor (holds sealed copy + relabels agents A–H): ______________________  Date: __________

_Any hyperparameter, feature-set, or training-data change after this timestamp voids the seal._
