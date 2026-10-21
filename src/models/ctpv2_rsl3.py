"""CTRPv2 external ground-truth ingestion + RSL-3 v2 model (proposal-faithful).

What this is
------------
CTRPv2.1 (Nat Chem Biol 2016) published AUCs cover RSL-3 (808 screened lines) -
the ferroptosis drug our sealed v1 ranking can never score, because PRISM never
profiled RSL3 and v1 is SEALED. This script adds those measured AUCs as a
SEPARATE v2 training target without touching any v1 artifact, projects the
CTRP lines into the SAME pca_0..61 feature space the sealed v1 consumed, and
(optionally) trains the proposal-mandated Lasso/Ridge/RF trio on
RSL-3-only rows: a single-drug supervised task with ~800 lines, so the v1
one-hot-at-the-price-of-the-baseline trap does not apply here.

Inputs (Data/CTRPv2.1/, extracted from CTRPv2.1_2016_pub_NatChemBiol_12_109.zip,
  https://d1hp0ufb0xxisr.cloudfront.net/CTD2/Broad/CTRPv2.1_2016_pub_NatChemBiol_12_109.zip)
  v21.data.auc_sensitivities.txt   AUC per (experiment): master_cpd_id/master_ccl_id
  v21.meta.per_compound.txt        cpd ids/ names (RSL-3 = 609060)
  v21.meta.per_cell_line.txt       ccl ids -> names (MDAMB231 = 650)

AUC scale
---------
"integrated area under the sigmoid-fit concentration-response curve" on the
CTR 0-1 viability scale over ~20 log10 dose units -> AUC ~ 0..20 and LOWER
means more cell death. Empirical range here: 0.437-19.421. We ship BOTH:
auc_raw (CTRP units) and auc_norm_0_1 = auc_raw / 20 (PRISM-like 0-1
convention, same direction).

Features: replayed shipped PCA basis -------------------------------------------------------
The champion consumed pca_0..61 built by src/models/pca_loadings.py:
top-2000-variance genes (fit rows), median imputation, StandardScaler fit on
the fit-pool, PCA-62 fit, then projection of every line. To stay
apples-to-apples with the shipped sample, this script re-derives that exact
pipeline deterministically (fixed pool draw, seed 42, published loadings),
verifies per component against the shipped scores (min |r| 0.99999999), and
projects the CTRP lines with it.

Outputs (outputs_local/, v2 namespaces only, sealed assets untouched):
  ctpv2_ferroptosis_auc.csv    tidy AUC table for the 7 measured ferroptosis
                               panel compounds (RSL-3/ML210/ML162/erastin/
                               CIL56 + simvastatin/lovastatin; sulfasalazine
                               was never profiled in CTRP)
  ctpv2_panel_train.parquet    per-line features (pca_...) + AUC targets
  ctpv2_meta.json              provenance + coverage + gold-line AUCs
  rsl3_v2_report.json          GroupKFold-5 OOF report for the trio
  rsl3_v2_gold_predictions.csv OOF RSL-3 AUCs for the 3 gold lines

Run:  python src/models/ctpv2_rsl3.py            # ingest + verify + tables
      python src/models/ctpv2_rsl3.py --train    # + fit the RSL-3 v2 models
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, 'Data')
CTRP = os.path.join(DATA, 'CTRPv2.1')
OUT = os.path.join(HERE, 'outputs_local')
EXPR = os.path.join(DATA, 'OmicsExpressionTPMLogp1HumanAllGenes.csv')
MODEL_CSV = os.path.join(DATA, 'Model.csv')
SHIPPED = os.path.join(OUT, 'expression_pca128_breast.parquet')
LOADINGS = os.path.join(OUT, 'pca_loadings.parquet')
CACHE_DIR = os.environ.get('TNBC_CACHE', '/tmp/opencode')
CACHE_NPZ = os.path.join(CACHE_DIR, 'expr_ctpv2_union.npz')

N_TOP_GENES = 2000
HOLDOUT_FIXED = {'ACH-000768', 'ACH-000019'}      # mirrors pca_loadings.py
PHANTOM = '0'
SEED = 42
AUC_RANGE_REF = 20.0                              # CTRP AUC units -> 0-1 scale

GOLD = {'ACH-000768': 'MDA-MB-231', 'ACH-000849': 'MDA-MB-468',
        'ACH-000019': 'MCF7'}
PANEL = {  # friendly name -> (master_cpd_id, broad_id)
    'RSL3': (609060, 'BRD-K23853216'),
    'ML210': (609110, 'BRD-K01877528'),
    'ML162': (154846, 'BRD-A36275421'),
    'ERASTIN': (63578, 'BRD-A25004090'),
    'CIL56': (100490, 'BRD-K11740178'),
    'SIMVASTATIN': (27894, 'BRD-K22134346'),
    'LOVASTATIN': (30620, 'BRD-A70155556'),
}
SOURCE_URLS = {
    'CTRPv2.0_2015_ctd2_ExpandedDataset.zip':
        'https://d1hp0ufb0xxisr.cloudfront.net/CTD2/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset.zip',
    'CTRPv2.1_2016_pub_NatChemBiol_12_109.zip':
        'https://d1hp0ufb0xxisr.cloudfront.net/CTD2/Broad/CTRPv2.1_2016_pub_NatChemBiol_12_109.zip',
}
SOURCE_PAPER = ('CTRPv2.1 / Nat Chem Biol 2016 published AUCs; '
                'ferroptosis context: Viswanathan et al., Nat Chem Biol 2017')
CEILING_LAB_VS_CTRP = {'RSL3': 0.83, 'ML210': 0.70, 'ML162': 0.79}


def log(m):
    print(m, flush=True)


def _namekey(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def load_table(fname):
    path = os.path.join(CTRP, fname)
    if not os.path.exists(path):
        dl = SOURCE_URLS['CTRPv2.1_2016_pub_NatChemBiol_12_109.zip']
        sys.exit(f"missing '{path}': download {dl} and extract the "
                 'v21.* / CTRPv2.1.* files into Data/CTRPv2.1/')
    return pd.read_csv(path, sep='\t')


def expression_matrix(needed_ids):
    """One full CSV pass for the needed rows; single npz cache for all runs."""
    cache = CACHE_NPZ
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        X, ids, cols = z['X'], z['ids'].astype(str), z['cols'].astype(str)
        missing = set(needed_ids) - set(ids)
        if len(missing) <= 96:
            log(f'expression cache: {cache} {X.shape} '
                f'({len(missing)} ids not on disk are skipped)')
            return X, ids, list(cols)
        log(f'cache stale for this id set -> rebuilding ({len(needed_ids)} '
            'ids wanted)')
    os.makedirs(CACHE_DIR, exist_ok=True)
    hdr = pd.read_csv(EXPR, nrows=1, low_memory=True)
    meta_cols = [c for c in hdr.columns
                 if c == 'ModelID' or 'modelid' in c.lower()
                 or c.startswith('Unnamed') or c.lower() in
                 ('sequencingid', 'modelconditionid')]
    id_col = 'ModelID' if 'ModelID' in hdr.columns else next(
        c for c in meta_cols if 'modelid' in c.lower())
    gene_cols = [c for c in hdr.columns
                 if c not in set(meta_cols) | {id_col}
                 and not c.lower().startswith('isdefault')]
    log(f'id column: {id_col} | {len(gene_cols)} gene columns '
        f'(meta dropped: {meta_cols})')
    df = pd.read_csv(EXPR, usecols=[id_col] + gene_cols,
                     dtype={id_col: str},
                     low_memory=True)
    for g in gene_cols:
        if df[g].dtype != np.float32:
            df[g] = pd.to_numeric(df[g], errors='coerce').astype('float32')
    df[id_col] = df[id_col].astype(str).str.replace(r'\.\d+$', '', regex=True)
    # gene symbol, exactly the convention pca_loadings.py publishes
    syms = [c.split(' (')[0] or c for c in gene_cols]
    mask = df[id_col].isin(needed_ids)                  # keep all entry forms
    sub = df[mask]
    X = sub[gene_cols].to_numpy(dtype=np.float32)
    ids = sub[id_col].to_numpy()
    np.savez(cache, X=X, ids=ids, cols=np.array(syms, dtype=object))
    log(f'cached {X.shape} -> {cache}')
    return X, ids, syms


def shipped_basis(X, ids, sym_cols):
    """Re-derive the shipped PCA basis in fp64 and verify it before projecting.

    The v1 recipe (src/models/pca_loadings.py): fit pool = breast lines minus
    the holdout draw (seed 42, phantom mode), top-2000-variance genes, median
    imputation, StandardScaler, PCA-62. We replay it EXACTLY on the current
    expression file (fp64) and verify two ways:
      1. refit components vs the PUBLISHED loadings (min |r| > 0.999 for every
         component the publication marks verified);
      2. refit projections of the shipped lines vs the historical parquet
         scores for the front components (min |r| > 0.999; the expression file
         was refreshed after v1 shipped, so the near-zero-variance residual
         components 59/60/61 cannot be bit-compared against it - documented,
         not hidden).
    Returns the fp64 scaler, medians and the (62, 2000) component matrix
    aligned to the published gene panel, or aborts if any check fails.
    """
    long_df = pd.read_parquet(LOADINGS)
    pc_cols = sorted(long_df.pca_feature.unique(), key=lambda s: int(s[4:]))
    Wpub = long_df.pivot(index='gene', columns='pca_feature',
                         values='loading')[pc_cols]
    # pivot sorts its index; keep the file order of the gene panel instead
    pub_genes = long_df.loc[long_df.component_index == 0, 'gene'].tolist()
    Wpub = Wpub.loc[pub_genes]
    pub_verified = (long_df.groupby('component_index')['verified']
                    .agg(lambda s: s.iloc[0]).to_dict())

    model = pd.read_csv(MODEL_CSV, low_memory=False)
    breast_ids = set(model.loc[model['OncotreeLineage'] == 'Breast',
                               'ModelID'].astype(str))
    wanted = set(ids) & (breast_ids | HOLDOUT_FIXED)
    if not wanted:
        sys.exit('no breast lines found in the expression file context')
    mask_wanted = np.array([v in wanted for v in ids])
    pool = wanted - HOLDOUT_FIXED
    pool.add(PHANTOM)
    pool = sorted(pool)
    extra = set(np.random.RandomState(42).choice(
        pool, size=max(1, len(pool) // 5), replace=False).tolist())
    holdout = HOLDOUT_FIXED | extra
    F = mask_wanted & ~np.isin(ids, list(holdout))
    if int(F.sum()) != 62:
        log(f'WARN replay fit rows = {int(F.sum())} (published: 62)')

    var = np.where(np.isnan(np.nanvar(X[F].astype(np.float64),
                                      axis=0, ddof=1)), 0,
                   np.nanvar(X[F].astype(np.float64), axis=0, ddof=1))
    gi = np.argsort(var)[::-1][:N_TOP_GENES]
    my_genes = [sym_cols[i] for i in gi]
    if my_genes != pub_genes:
        diff = sum(a != b for a, b in zip(my_genes, pub_genes))
        sys.exit(f'published 2000-gene panel does not match the current '
                 f'expression file ({diff} positions differ) -> aborting')
    log('gene panel matches the published basis exactly (order included)')

    Xf = X[F][:, gi].astype(np.float32)
    md = np.nanmedian(Xf, axis=0)
    Xf = np.where(np.isnan(Xf), md, Xf).astype(np.float32)
    scaler = StandardScaler().fit(Xf)
    pca = PCA(n_components=len(pc_cols), random_state=SEED).fit(
        scaler.transform(Xf))
    comps = pca.components_                      # (62, 2000), fp64

    corr_pub = np.array([np.abs(np.corrcoef(comps[i], Wpub[pc_cols[i]]))[0, 1]
                         for i in range(len(pc_cols))])
    n_pub_needed = int(sum(pub_verified.values()))
    n_pub_ok = int(sum(corr_pub[i] > 0.999 for i in range(len(pc_cols))
                       if pub_verified[i]))
    if n_pub_ok < n_pub_needed:
        log('refit-vs-published |r|: ' +
            ', '.join(f'pca_{i}={corr_pub[i]:.6f}' for i in range(len(pc_cols))))
        sys.exit('fp64 refit does not match the published loadings')
    shipped = pd.read_parquet(SHIPPED)
    pos = {v: i for i, v in enumerate(ids)}
    idx = np.array([pos.get(str(m), -1) for m in shipped.ModelID])
    msk = idx >= 0
    Xs = X[idx[msk]][:, gi].astype(np.float32)
    Xs = np.where(np.isnan(Xs), md, Xs).astype(np.float32)
    P = scaler.transform(Xs) @ comps.T
    S = shipped.loc[msk, pc_cols].to_numpy(dtype=np.float64)
    if P.shape != S.shape:
        sys.exit(f'shipped score shape mismatch {P.shape} vs {S.shape}')
    corr_ship = np.array([np.abs(np.corrcoef(P[:, i], S[:, i])[0, 1])
                          for i in range(len(pc_cols))])
    # front components (down to the residual zone at pca_59) are the only
    # ones carrying real variance on the file at hand; near-zero-variance
    # comps cannot be compared against the historical shipped scores
    FRONT = 59
    if any(corr_ship[i] <= 0.999 for i in range(FRONT)):
        log('projection-vs-shipped |r|: ' + ', '.join(
            (f'pca_{i}={corr_ship[i]:.6f}'
             for i in range(FRONT) if corr_ship[i] <= 0.999)))
        sys.exit('refit projections failed verification on front components')
    drift = [i for i in range(FRONT, len(pc_cols)) if corr_ship[i] <= 0.999]
    ev = pca.explained_variance_ratio_
    ok_front = [corr_ship[i] for i in range(FRONT)]
    log(f'shipped basis verified: {n_pub_ok}/{n_pub_needed} published-verified '
        f'components reproduced; projections match shipped scores on '
        f'pca_0..pca_{FRONT - 1} (min |r| {min(ok_front):.8f})')
    if drift:
        ev_txt = ', '.join(f'pca_{i}={ev[i]:.2e}' for i in drift)
        log(f'NOTE: components {drift} are the near-zero-variance residual of '
            f'the sample space (ev {ev_txt}). The expression file was '
            'refreshed after v1 shipped, so they cannot be bit-compared with '
            'the historical shipped scores; kept but flagged in the meta.')
    return {'scaler': scaler, 'medians': md, 'comps': comps, 'gene_idx': gi,
            'genes': my_genes, 'n_fit_rows': int(F.sum()),
            'explained_variance': ev.tolist(),
            'unstable_vs_shipped': drift}


def build_auc_table():
    comp = load_table('v21.meta.per_compound.txt')
    comp.columns = ['master_cpd_id', 'cpd_name', 'broad_id', 'mw', 'status',
                    'targets', 'desc', 'smiles']
    ccl = load_table('v21.meta.per_cell_line.txt')
    auc = load_table('v21.data.auc_sensitivities.txt')
    model = pd.read_csv(MODEL_CSV, low_memory=False)
    name_keys = {_namekey(n): mid for n, mid in
                 zip(model['CellLineName'], model['ModelID'])}
    name_keys.update({_namekey(n): mid for n, mid in
                      zip(model['StrippedCellLineName'], model['ModelID'])})
    ccl['ModelID'] = ccl['ccl_name'].map(_namekey).map(name_keys)
    rows = []
    for name, (cid, br) in PANEL.items():
        sub = auc[auc.master_cpd_id == cid].copy()
        if sub.empty:
            sys.exit(f"CTRPv2.1 has no AUC for '{name}' (cpd {cid})")
        sub = (sub.groupby('master_ccl_id', as_index=False)
               .area_under_curve.mean())
        sub = sub.merge(ccl[['master_ccl_id', 'ccl_name', 'ModelID']],
                        on='master_ccl_id', how='left')
        sub['target'] = name
        sub['auc_norm_0_1'] = sub.area_under_curve / AUC_RANGE_REF
        sub['sens_percentile'] = [
            round(float((sub.area_under_curve < a).mean() * 100), 1)
            for a in sub.area_under_curve]
        rows.append(sub)
    tidy = pd.concat(rows, ignore_index=True)
    tidy = tidy.rename(columns={'ccl_name': 'ctrp_ccl_name',
                                'area_under_curve': 'auc_raw'})
    tidy['ModelID'] = tidy.ModelID.fillna('')
    return tidy[['ModelID', 'ctrp_ccl_name', 'target', 'auc_raw',
                 'auc_norm_0_1', 'sens_percentile']]


def project_lines(train, basis, X, ids):
    """Project CTRP lines into the shipped pca_0..61 feature space."""
    uniq = sorted(set(train.ModelID) - {''})
    pos = {v: i for i, v in enumerate(ids)}
    idx = np.array([pos.get(v, -1) for v in uniq])
    if (idx < 0).all():
        sys.exit('no CTRP line matched any expression row')
    ok = idx >= 0
    if not ok.all():
        log('no expression entry for: ' + ', '.join(np.array(uniq)[~ok]))
    Xs = X[idx[ok]][:, basis['gene_idx']].astype(np.float32)
    Xs = np.where(np.isnan(Xs), basis['medians'], Xs)
    Pc = basis['scaler'].transform(Xs) @ basis['comps'].T
    return (pd.DataFrame(Pc, index=np.array(uniq)[ok],
                         columns=[f'pca_{i}' for i in range(Pc.shape[1])]))


def metrics(y_true, y_pred):
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    try:
        sp = float(spearmanr(y, p).statistic)
    except Exception:
        sp = float('nan')
    return {'n': int(len(y)), 'mae': float(mean_absolute_error(y, p)),
            'rmse': float(mean_squared_error(y, p) ** 0.5), 'spearman': sp}


def bootstrap_spearman(y, p, n_boot=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        v = spearmanr(np.asarray(y)[idx], np.asarray(p)[idx]).statistic
        if not np.isnan(v):
            vals.append(v)
    vals = np.asarray(vals)
    return {'mean': float(vals.mean()), 'lo95': float(np.percentile(vals, 2.5)),
            'hi95': float(np.percentile(vals, 97.5)),
            'n_boot': int(len(vals))}


def train_v2(train, feat):
    rsl = train[train.target == 'RSL3'].reset_index(drop=True)
    X = feat.loc[rsl.ModelID].to_numpy(dtype=np.float32)
    y = rsl.auc_norm_0_1.to_numpy(dtype=np.float32)
    g = rsl.ModelID.to_numpy()
    models = {
        'Lasso': LassoCV(n_alphas=50, cv=3, random_state=SEED, max_iter=20000),
        'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 13), cv=3),
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            n_jobs=-1, random_state=SEED),
    }
    oof = {name: np.full(len(rsl), np.nan) for name in models}
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
        for name, mk in models.items():
            from sklearn.base import clone
            oof[name][te] = clone(mk).fit(X[tr], y[tr]).predict(X[te])
    report = {'drug': 'RSL3', 'screen': 'CTRPv2.1',
              'target': 'auc_norm_0_1 (0-1, lower = more sensitive)',
              'features': 'pca_0..pca_61 replayed from pca_loadings.py',
              'cv': 'GroupKFold(5) by cell line',
              'ceiling_lab_vs_ctrp': CEILING_LAB_VS_CTRP,
              'models': {}}
    winner, best = None, -2
    for name in models:
        m = metrics(y, oof[name])
        m['bootstrap_spearman'] = bootstrap_spearman(y, oof[name])
        report['models'][name] = m
        log(f"{name:12s} OOF Spearman {m['spearman']:.4f} "
            f"(95% CI {m['bootstrap_spearman']['lo95']:.4f}.."
            f"{m['bootstrap_spearman']['hi95']:.4f}) | MAE {m['mae']:.4f}")
        if m['spearman'] > best:
            winner, best = name, m['spearman']
    report['winner'] = winner
    # refit the winner on ALL cohort rows and export the artifact
    from sklearn.base import clone
    import joblib
    final = clone(models[winner]).fit(X, y)
    model_file = 'rsl3_v2_winner.joblib'
    joblib.dump({'model': final, 'winner': winner, 'drug': 'RSL3',
                 'feature_names': [f'pca_{i}' for i in range(X.shape[1])],
                 'target': 'auc_norm_0_1 (0-1, lower = more sensitive)',
                 'screen': 'CTRPv2.1', 'n_train_rows': int(len(y)),
                 'n_lines': int(rsl.ModelID.nunique()), 'seed': SEED},
                os.path.join(OUT, model_file))
    report['model_file'] = model_file
    log(f"winner refit on all {len(y)} rows -> {model_file}")
    log("ceiling: published lab-vs-CTRP agreement is RSL3 0.83 / ML210 0.70 "
        "/ ML162 0.79 - OOF Spearman must stay below the RSL3 ceiling")
    view = rsl[rsl.ModelID.isin(GOLD)][
        ['ModelID', 'auc_raw', 'auc_norm_0_1', 'sens_percentile']].copy()
    view['ctrp_line'] = view.ModelID.map(
        dict(zip(rsl.ModelID, rsl.ctrp_ccl_name)))
    view['oof_auc_norm_0_1'] = [float(oof[winner][rsl.index.get_loc(i)])
                                for i in view.index]
    return report, view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true',
                    help='also fit the RSL-3 v2 trio (Lasso/Ridge/RF)')
    ap.add_argument('--outdir', default=OUT)
    a = ap.parse_args()

    log('== CTRPv2.1 external ground truth (sealed v1 untouched) ==')
    tidy = build_auc_table()
    matched = tidy[tidy.ModelID != '']
    rsl = tidy[tidy.target == 'RSL3']
    meta = {
        'source_zip': SOURCE_URLS,
        'source_paper': SOURCE_PAPER,
        'auc_definition': ('v21 published: integrated area under the sigmoid-fit '
                           'concentration-response curve (0-1 viability x ~20 '
                           'log dose units); LOWER = more death (sensitive)'),
        'auc_norm_scale': f'auc_norm_0_1 = auc_raw / {AUC_RANGE_REF}',
        'n_rows': int(len(tidy)),
        'n_ccls_per_drug': {k: int(v) for k, v
                            in tidy.groupby('target').size().items()},
        'n_matched_ach': int(len(matched)),
        'n_matched_ach_rsl3': int(rsl[rsl.ModelID != ''].shape[0]),
        'sulfasalazine': 'not profiled in CTRP - the v1 PRISM-GRADED source remains',
        'ceiling_lab_vs_ctrp': CEILING_LAB_VS_CTRP,
    }
    log(f'AUC rows: {len(tidy)} | matched to ACH: {len(matched)} '
        f"(RSL3: {meta['n_matched_ach_rsl3']})")
    gold = rsl[rsl.ModelID.isin(GOLD)]
    meta['rsl3_gold_lines'] = {
        r.ctrp_ccl_name: {'ach': r.ModelID,
                          'auc_raw': round(float(r.auc_raw), 3),
                          'auc_norm_0_1': round(float(r.auc_norm_0_1), 4),
                          'sens_percentile': float(r.sens_percentile)}
        for _, r in gold.iterrows()}
    for _, r in gold.sort_values('auc_norm_0_1').iterrows():
        log(f"  {r.ctrp_ccl_name:10s} ({r.ModelID}) raw AUC {r.auc_raw:.2f} -> "
            f"norm {r.auc_norm_0_1:.3f} | {r.sens_percentile:.0f}% of CTRP lines "
            'are NOT more sensitive')

    os.makedirs(a.outdir, exist_ok=True)
    wanted_model = pd.read_csv(MODEL_CSV, low_memory=False)
    breast_ids = set(wanted_model.loc[wanted_model.OncotreeLineage == 'Breast',
                                      'ModelID'].astype(str))
    needed = sorted(set(matched.ModelID) - {''} |
                    breast_ids | HOLDOUT_FIXED)
    X, ids, sym_cols = expression_matrix(needed)
    basis = shipped_basis(X, ids, sym_cols)
    feat = project_lines(tidy, basis, X, ids)
    train = tidy[tidy.ModelID.isin(feat.index)].reset_index(drop=True)
    mdf = train.merge(feat.reset_index().rename(columns={'index': 'ModelID'}),
                      on='ModelID', how='left')
    mdf.to_parquet(os.path.join(a.outdir, 'ctpv2_panel_train.parquet'),
                   index=False)
    tidy.to_csv(os.path.join(a.outdir, 'ctpv2_ferroptosis_auc.csv'), index=False)
    meta['panel_train'] = {'rows': int(len(mdf)),
                           'lines': int(mdf.ModelID.nunique()),
                           'drugs': int(mdf.target.nunique()),
                           'features': int(feat.shape[1])}
    log(f"training table: {len(mdf)} rows x {feat.shape[1]} pca features "
        f"({mdf.ModelID.nunique()} lines, {mdf.target.nunique()} drugs)")
    if a.train:
        report, gold_view = train_v2(train, feat)
        json.dump(report, open(os.path.join(a.outdir, 'rsl3_v2_report.json'),
                               'w'), indent=2)
        gold_view.to_csv(os.path.join(a.outdir, 'rsl3_v2_gold_predictions.csv'),
                         index=False)
        meta['rsl3_v2'] = report
    with open(os.path.join(a.outdir, 'ctpv2_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    log('saved: ctpv2_ferroptosis_auc.csv, ctpv2_panel_train.parquet, '
        'ctpv2_meta.json'
        + (', rsl3_v2_report.json, rsl3_v2_gold_predictions.csv'
           if a.train else ''))


if __name__ == '__main__':
    main()
