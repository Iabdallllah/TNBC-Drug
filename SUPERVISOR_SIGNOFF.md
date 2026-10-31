# SUPERVISOR SIGNOFF — v1.1 defense (v1.2 tag)
Status: NOT SIGNED (fields blank per rules — do not invent names/signatures/dates).

## What is sealed (unchanged)
- sealed_prediction.md (SHA256 de0922e70c97b416eb46bc9df73a417dfa959551f74cb2043e543dba4f4db851)
- v1.0-defense tag (commit 8be8319)
- v1.1-defense tag (commit bb86245); v1.2-defense (current) only adds docs/infra/diagnostic, not the sealed artifact

## What v1.1 changed (addendum only, not the sealed prediction)
- Stale-claim sweep; baseline audit (per-drug mean beats CatBoost: Spearman −0.199, R² −0.257); selectivity null survives (Spearman 0.029, n=66); 76 unique expression lines (not 77); 55/55 AUC<0.3 in learned 227 (coverage, not lift); Exatecan measured Δ = −0.1399 (opposite sign); NDCG graded relevance corrected.
- No edit to catboost_tnbc_v1.cbm, leaderboard.csv, v1.0 rankings.

## What is claimed and NOT claimed
- CLAIMED (post-hoc, exploratory): baseline comparison; null selectivity; 76-line correction; zero-JS report; 37/37 tests.
- NOT CLAIMED: validated predictor; validated selective drugs; expression contribution proven; blind design fully validated; any panel kill percentage as an absolute rate.

## Blinding scope (Phase 4.1 fix-forward)
- UI blinding (BLIND_MODE ON) hides: panel predictions table, FerroSelect panel ranking, kill proxies, and the Live predictor for panel agents in both the dashboard and the exported Defence report; a banner replaces them. AppTest covers ON/OFF.
- UI blinding does NOT hide committed files. The following files remain in the repo and must be withheld from the bench team out-of-band:
  - outputs_local/sealed_prediction.md (+ sealed_prediction_manifest.json) — sealed record, SHA de0922e7…4db851
  - outputs_local/leaderboard.csv, outputs_local/ferroselect_panel_predictions.csv, outputs_local/mda_mb_231_full_ranking.csv, outputs_local/candidate_rankings.csv — sealed/panel numbers
  - catboost_tnbc_v1.cbm, outputs_local/prediction_intervals.json, outputs_local/ctpv2_* — model artifacts
- The bench team must receive only the protocol / plate map and the blinded dashboard until unblinding is authorised.

## Checklist (all blank until supervisor completes)
- [ ] Supervisor name / role: ___________________________
- [ ] Signature (text or digital): ______________________
- [ ] Date (UTC): _______________________________  (before plate 1; actual date by supervisor)
- [ ] Confirmed sealed artifacts untouched at v1.0 and v1.1: [ ]
- [ ] Confirmed dashboard-access log entries will be made by supervisor (not by agent) before plate 1: [ ]
- [ ] Decision: keep diagnostic on slides? (YES/NO — requires supervisor): _____
- [ ] BLIND_MODE set ON; UNBLIND path documented; banner visible: [ ]

Requires a human: supervisor sign-off, actual access-log entries before plate 1, and the diagnostic-slide decision.
