# Diagnostic pre-registration (pre-run commit)

Status: PRE-REGISTERED (this file's commit timestamp is the preregistration).
Question (post-hoc, exploratory, 23 training lines only; 7 holdout lines NOT touched):
does expression-derived line similarity improve within-line drug ranking over a
per-drug mean, estimated without using the 7 holdout lines?

Data / folds: only the 23 training cell lines (exclude MDA-MB-231 ACH-000768 and MCF7 ACH-000019 plus any other holdout set; the full ML set has 30 lines of which 7 are holdout, so 23 train). GroupKFold(5) by cell line, fixed random_state=42, identical across M0..M3.

Models (all computed on training folds only; per-drug means / PCA offsets / neighbours computed inside the folds):
 M0: per-drug mean (global-mean fallback for unseen drugs), computed per fold on training lines.
 M1: M0 plus per-line offset from 62 PCA features predicted by RidgeCV (alpha grid). Offset = RidgeCV(PCA_features_train) -> residual over M0; applied to held-out line by its PCA vector.
 M2: M0 plus kNN over lines in PCA space (pre-declared k=5, Euclidean on first 10 PCs, inverse-distance weights, borrow neighbouring lines' drug-wise M0 residuals). No adaptive k.
 M3 (reference): CatBoost Regressor (same settings as v1.0, depth=8, iterations=800, learning_rate=0.05) on the same folds if runtime <30 min per fold else skipped and noted.

Metrics (per held-out line within each fold; then pooled):
- per-line Spearman (mean across held-out lines); pooled R2 (over held-out pairs); MAE; NDCG@10 graded rel=1-AUC; Recall@10 binary AUC<0.3.
- paired bootstrap over HELD-OUT LINES (2,000 resamples) for each difference vs M0: compute per-line Spearman difference, resample held-out lines, report 95% CI.
- M1 construction means its per-line Spearman equals M0 by design; assert numerically and judge M1 only on pooled R2 / MAE.
- M2 "success" only if paired per-line Spearman difference vs M0 has 95% CI above 0. Otherwise "no evidence of expression signal" (not proof of absence).
- NDCG NOT used for any claim.

Success criteria (fixed in advance):
1. M1 Spearman == M0 (assert; judge on R2/MAE).
2. M2 Spearman CI>0 required to claim expression signal; otherwise no claim.
3. Report whatever the numbers are (null / loss / wide CI is honest).
4. All new results labelled "post-hoc, exploratory".
5. No holdout line ModelIDs appear in diagnostic outputs.

Limits / exclusions: no edit to catboost_tnbc_v1.cbm; no 7-holdout use; no unregistered metric additions.
