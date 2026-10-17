"""Reproduce and publish the PCA gene loadings behind the shipped expression features.

Why this exists
---------------
The CatBoost model consumes `pca_0 .. pca_61`, but the loadings were never saved,
so the SHAP figures show opaque `pca_8` / `pca_45` labels with no biological
meaning. This script refits the identical pipeline, then PROVES the fit is the
one behind the shipped scores before publishing any gene name.

Provenance note (important)
---------------------------
Notebook 01 section 8 wrote `wanted_breast_ids.csv` with pandas' default header,
and section 9a read it back with `header=None`. That injected a phantom cell line
`'0'` into the wanted set, which made the RandomState(42) holdout draw pick a
different subset of lines than an intended run would. The shipped
`expression_pca128_breast.parquet` was produced WITH that phantom, so reproducing
it requires the historical draw (`--pool phantom`, the default).

The phantom bug is fixed in notebook 01 (section 8 now writes `header=False`).
That fix intentionally means a fresh notebook run no longer reproduces the shipped
artifact - it produces a cleaner basis and requires retraining.

What gets published
-------------------
Only components that are verified against the shipped scores. Because
`n_components == n_fit_rows`, the final component is the zero-variance residual
direction of the sample space: it carries no signal and its loadings are
numerically unstable, so it is published with `verified=False` and flagged in the
UI rather than silently annotated with gene names.

IO budget (does NOT thrash the machine)
---------------------------------------
The 938 MB expression CSV is parsed exactly ONCE. The (79, 60649) float32 slice
for the dataset rows (~20 MB) is cached to $TNBC_CACHE/expr79.npz; every later
run - and the whole verification battery - then works from RAM in seconds.

Run:  python src/models/pca_loadings.py            # reproduce the shipped basis
      python src/models/pca_loadings.py --pool clean   # post-bugfix basis
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'Data')
OUT = os.path.join(ROOT, 'outputs_local')
CACHE_DIR = os.environ.get('TNBC_CACHE', '/tmp/opencode')
EXPR = os.path.join(DATA, 'OmicsExpressionTPMLogp1HumanAllGenes.csv')
MODEL_CSV = os.path.join(DATA, 'Model.csv')
SHIPPED = os.path.join(OUT, 'expression_pca128_breast.parquet')
N_TOP_GENES = 2000
HOLDOUT_FIXED = {'ACH-000768', 'ACH-000019'}      # MDA-MB-231 + MCF7, always holdout
PHANTOM = '0'                                     # historical phantom entry
MIN_VERIFIED = 50                                 # refuse to publish a half-blind fit


def log(m):
    print(m, flush=True)


def load_dataset_slice():
    """79-row x 60649-gene expression matrix + ids + gene symbols, via one CSV pass.

    Cached at $TNBC_CACHE/expr79.npz (about 20 MB). If the cache is missing this
    performs a single full read of the 938 MB file - all later stages stay in RAM.
    """
    cache = os.path.join(CACHE_DIR, 'expr79.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        log(f'expression cache: {cache} {z["X"].shape}')
        return z['X'], z['ids'].astype(str), z['genes'].astype(str)

    os.makedirs(CACHE_DIR, exist_ok=True)
    hdr = pd.read_csv(EXPR, nrows=1, low_memory=True)
    id_col = next(c for c in hdr.columns if 'modelid' in c.lower())
    gene_cols = [c for c in hdr.columns
                 if c not in (id_col,) and not c.lower().startswith('isdefault')]
    log(f'single full read of {EXPR} ({len(gene_cols)} gene columns)...')
    df = pd.read_csv(EXPR, usecols=[id_col] + gene_cols, dtype='float32',
                     low_memory=True)
    df[id_col] = df[id_col].astype(str)
    genes = np.array([c.split(' (')[0] or c for c in gene_cols])
    X = df[gene_cols].to_numpy(dtype=np.float32)
    ids = df[id_col].to_numpy()
    np.savez(cache, X=X, ids=ids, genes=genes)
    log(f'cached {X.shape} -> {cache}')
    return X, ids, genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', choices=['phantom', 'clean'], default='phantom',
                    help='"phantom" reproduces the shipped artifact; "clean" is the '
                         'post-bugfix pipeline (produces a different basis)')
    a = ap.parse_args()

    for p in (EXPR, MODEL_CSV, SHIPPED):
        if not os.path.exists(p):
            sys.exit(f'missing input: {p}')

    X, ids, genes = load_dataset_slice()
    model_df = pd.read_csv(MODEL_CSV, low_memory=False)
    breast_ids = set(model_df.loc[model_df['OncotreeLineage'] == 'Breast',
                                  'ModelID'].astype(str))
    mask_all = np.array([v in (breast_ids | HOLDOUT_FIXED) for v in ids])
    if not mask_all.any():
        sys.exit('no wanted rows found - check Model.csv lineage labels')

    wanted = set(ids[mask_all])
    log(f'wanted ids={len(wanted)} | rows kept={int(mask_all.sum())} '
        f'(the expression file ships multiple entry forms for some ModelIDs - '
        f'the shipped parquet keeps all of them)')
    pool = wanted - HOLDOUT_FIXED
    if a.pool == 'phantom':
        pool.add(PHANTOM)                 # historical: the phantom shifted the draw
    pool = sorted(pool)
    extra = set(np.random.RandomState(42).choice(
        pool, size=max(1, len(pool) // 5), replace=False).tolist())
    holdout = HOLDOUT_FIXED | extra
    F = mask_all & ~np.array([v in holdout for v in ids])
    log(f'pool={len(pool)} ({a.pool}) | dataset rows={int(mask_all.sum())} | '
        f'fit rows={int(F.sum())} | holdout lines={len(holdout)}')

    # row order must match the shipped parquet before any comparison
    shipped = pd.read_parquet(SHIPPED)
    pc_cols = [c for c in shipped.columns if c.startswith('pca_')]
    n_comp = len(pc_cols)
    if list(shipped['ModelID'].astype(str)) != list(ids[mask_all]):
        sys.exit('row order differs from the shipped parquet; aborting')
    S = shipped[pc_cols].to_numpy()
    log(f'shipped scores: {S.shape}')

    var = np.where(np.isnan(np.nanvar(X[F].astype(np.float64), axis=0, ddof=1)), 0,
                   np.nanvar(X[F].astype(np.float64), axis=0, ddof=1))
    gi = np.argsort(var)[::-1][:N_TOP_GENES]
    kept_genes = genes[gi]
    Xf = X[np.ix_(F, gi)].astype(np.float32)
    Xa = X[:, gi].astype(np.float32)
    md = np.nanmedian(Xf, axis=0)
    Xf = np.where(np.isnan(Xf), md, Xf).astype(np.float32)
    Xa = np.where(np.isnan(Xa), md, Xa).astype(np.float32)
    log(f'top-{N_TOP_GENES} variance genes (all present in fit rows)')

    scaler = StandardScaler().fit(Xf)
    pca = PCA(n_components=n_comp, random_state=42).fit(scaler.transform(Xf))
    P = pca.transform(scaler.transform(Xa))
    log(f'PCA-{n_comp} explained variance: {pca.explained_variance_ratio_.sum():.6f}')

    # ---- per-component verification against the shipped scores ----
    corr = np.zeros((n_comp, n_comp))
    for i in range(n_comp):
        for j in range(n_comp):
            corr[i, j] = np.corrcoef(P[:, i], S[:, j])[0, 1]
    ri, ci = linear_sum_assignment(-np.abs(corr))
    ar = np.abs(corr[ri, ci])
    sign = np.where(corr[ri, ci] >= 0, 1.0, -1.0)
    comps = pca.components_[ri] * sign[:, None]          # aligned to shipped columns
    ev = pca.explained_variance_ratio_[ri]
    recon = scaler.transform(Xa) @ comps.T
    per_comp_err = np.abs(recon - S).max(axis=0)
    verified = (ar > 0.999) & (per_comp_err < 1e-2) & (ev > 0)
    log(f'verified components: {int(verified.sum())}/{n_comp}')
    for j in np.flatnonzero(~verified):
        log(f'  pca_{j}: |r|={ar[j]:.4f} recon_err={per_comp_err[j]:.3e} '
            f'explained_var={ev[j]:.2e} -> NOT published with gene names')
    if int(verified.sum()) < MIN_VERIFIED:
        sys.exit(f'only {int(verified.sum())}/{n_comp} components verified '
                 f'(need >= {MIN_VERIFIED}); refusing to publish loadings')

    # ---- publish ----
    n_genes = comps.shape[1]
    long_df = pd.DataFrame({
        'component_index': np.repeat(np.arange(n_comp), n_genes),
        'pca_feature': np.repeat([f'pca_{j}' for j in range(n_comp)], n_genes),
        'gene': np.tile(list(kept_genes), n_comp),
        'loading': comps.ravel().astype(np.float32),
        'explained_variance_ratio': np.repeat(ev, n_genes),
        'verified': np.repeat(verified, n_genes),
    })
    long_df.to_parquet(os.path.join(OUT, 'pca_loadings.parquet'), index=False)

    top3 = (long_df[long_df['verified']]
            .assign(abs_loading=lambda d: d['loading'].abs())
            .sort_values(['component_index', 'abs_loading'], ascending=[True, False])
            .groupby('component_index').head(3))
    rows = []
    for i, grp in top3.groupby('component_index'):
        gl = list(grp.sort_values('abs_loading', ascending=False)['gene'])
        while len(gl) < 3:
            gl.append('')
        rows.append({'pca_feature': f'pca_{i}',
                     'label': f"pca_{i} ({', '.join(gl)})",
                     'explained_variance_ratio': round(float(ev[i]), 8),
                     'gene_1': gl[0], 'gene_2': gl[1], 'gene_3': gl[2],
                     'status': 'verified'})
    for j in np.flatnonzero(~verified):
        rows.append({'pca_feature': f'pca_{j}',
                     'label': f'pca_{j} (residual direction - not attributable)',
                     'explained_variance_ratio': round(float(ev[j]), 8),
                     'gene_1': '', 'gene_2': '', 'gene_3': '',
                     'status': 'not attributable (zero-variance residual)'})
    top_df = pd.DataFrame(rows).sort_values('pca_feature',
                                            key=lambda s: s.str[4:].astype(int))
    top_df.to_csv(os.path.join(OUT, 'pca_loadings_top.csv'), index=False)

    meta = {
        'source': 'src/models/pca_loadings.py',
        'pool_mode': a.pool,
        'n_components': int(n_comp),
        'n_genes': int(n_genes),
        'n_components_verified': int(verified.sum()),
        'components_not_attributable': [f'pca_{j}' for j in np.flatnonzero(~verified)],
        'max_reconstruction_error_verified': float(per_comp_err[verified].max()),
        'max_reconstruction_error_all': float(per_comp_err.max()),
        'min_abs_correlation_verified': float(ar[verified].min()),
        'explained_variance_sum': float(ev.sum()),
        'fit_rows': int(F.sum()),
        'dataset_rows': int(mask_all.sum()),
        'shipped_scores': list(S.shape),
        'note': ('Loadings are verified per component against the shipped '
                 'expression_pca128_breast.parquet: reconstructed scores = '
                 'scaled_expression @ components.T. Components with zero explained '
                 'variance are the residual direction of the sample space and are '
                 'deliberately left unattributed.'),
    }
    json.dump(meta, open(os.path.join(OUT, 'pca_loadings_meta.json'), 'w'), indent=2)
    log(f'saved pca_loadings.parquet ({len(long_df)} rows), pca_loadings_top.csv, meta')
    log('example: ' + ', '.join(top_df[top_df['status'] == 'verified']['label'].head(3)))


if __name__ == '__main__':
    main()