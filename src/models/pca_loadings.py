"""Recompute PCA gene loadings for the frozen expression features.

Why this exists
---------------
The CatBoost model is trained on 62 principal components of the top-2000
variance genes, but notebook 01 never saved `pca.components_`, so the SHAP
figures show opaque `pca_8`, `pca_45` labels with no biological meaning.

This script REPRODUCES the exact fit that produced
`expression_pca128_breast.parquet` and exports the loadings, so every PC can be
annotated as `pca_8 (GENE1, GENE2, GENE3)`.

Reproducibility contract
------------------------
It mirrors notebook 01 cells §8/§9a/§9b (GENE_START auto-detected,
IsDefaultEntryForModel filter, RandomState(42) holdout, ddof=1 variance on
non-holdout rows, top-2000, <=20% missing, fit-row median impute,
StandardScaler then PCA(n_components=min(128, n_fit, n_genes))).

It deliberately does NOT apply the later dedup block: the shipped parquet has
79 rows / 62 PCs, and deduplicating would yield 60 PCs and silently break the
frozen 1,544-feature model.

The refit is verified against the shipped parquet before anything is written —
if the scores do not match, the script aborts instead of publishing wrong
gene attributions.

Run:  python src/models/pca_loadings.py
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'Data')
OUT = os.path.join(ROOT, 'outputs_local')
EXPR = os.path.join(DATA, 'OmicsExpressionTPMLogp1HumanAllGenes.csv')
MODEL_CSV = os.path.join(DATA, 'Model.csv')
SHIPPED = os.path.join(OUT, 'expression_pca128_breast.parquet')
CACHE_DIR = os.environ.get('TNBC_CACHE', '/tmp/opencode')
N_TOP_GENES = 2000
HOLDOUT_FIXED = {'ACH-000768', 'ACH-000019'}     # MDA-MB-231 + MCF7, always holdout
TOL = 1e-3                                       # score agreement needed to trust loadings


def log(msg):
    print(msg, flush=True)


def gene_symbol(header):
    """'TSPAN6 (7105)' -> 'TSPAN6'; keep Ensembl IDs as-is."""
    return header.split(' (')[0].strip() or header


def layout():
    with open(EXPR, newline='', encoding='utf-8', errors='replace') as f:
        hdr = next(csv.reader(f))
    low = [h.strip().lower() for h in hdr]
    id_pos = next(i for i, h in enumerate(low) if 'modelid' in h)
    def_pos = next((i for i, h in enumerate(low) if 'isdefaultentryformodel' in h), None)
    gene_start = max([0, id_pos] + ([def_pos] if def_pos is not None else [])) + 1
    with open(EXPR, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader)
        probe = next(reader)

    def _isnum(v):
        v = v.strip()
        if v == '' or v.lower() in {'na', 'nan', 'none', 'null'}:
            return True
        try:
            float(v)
            return True
        except ValueError:
            return False

    num_start = 0
    for i, v in enumerate(probe[:12]):
        if not _isnum(v):
            num_start = i + 1
    gene_start = max(gene_start, num_start)
    log(f'layout: id_col={hdr[id_pos]} gene_start={gene_start} '
        f'total_cols={len(hdr)} genes={len(hdr) - gene_start}')
    return hdr, id_pos, def_pos, gene_start


def main():
    for p in (EXPR, MODEL_CSV, SHIPPED):
        if not os.path.exists(p):
            sys.exit(f'missing input: {p}')
    hdr, id_pos, def_pos, gene_start = layout()
    id_col = hdr[id_pos]
    gene_cols = list(range(gene_start, len(hdr)))
    genes = [gene_symbol(hdr[i]) for i in gene_cols]

    model_df = pd.read_csv(MODEL_CSV, low_memory=False)
    breast_ids = set(model_df.loc[model_df['OncotreeLineage'] == 'Breast',
                                  'ModelID'].astype(str))

    use = sorted({id_pos} | ({def_pos} if def_pos is not None else set()))
    file_idx = pd.read_csv(EXPR, usecols=use, engine='c', low_memory=True)
    file_idx.columns = [hdr[i] for i in use]
    flag = file_idx[hdr[def_pos]] if def_pos is not None else pd.Series('1', index=file_idx.index)
    is_default = flag.astype(str).str.strip().str.lower().isin(
        {'true', 'yes', '1', 't', 'y'})
    expr_ids = set(file_idx.loc[is_default, id_col].astype(str))
    file_ids = file_idx[id_col].astype(str).to_numpy()

    # The shipped parquet kept every file row for a wanted line (79 rows, 76 unique:
    # MCF7 plus two other lines appear twice), i.e. no IsDefaultEntryForModel filter.
    # Applying that filter would drop to 76 rows / 60 PCs and break the frozen model.
    mask_all = np.array([v in (breast_ids | HOLDOUT_FIXED) for v in file_ids])
    wanted = sorted((breast_ids & expr_ids) | HOLDOUT_FIXED)
    pool = sorted(set(wanted) - HOLDOUT_FIXED)
    rng = np.random.RandomState(42)
    holdout = HOLDOUT_FIXED | set(rng.choice(pool, size=max(1, len(pool) // 5),
                                            replace=False).tolist())
    mask_var = mask_all & ~np.array([v in holdout for v in file_ids])
    log(f'wanted lines: {len(wanted)} | file rows: {int(mask_all.sum())} '
        f'| fit rows: {int(mask_var.sum())} | holdout lines: {len(holdout)}')

    # ---- variance scan on fit rows only (batched to bound memory) ----
    G = len(gene_cols)
    BATCH = 15000
    cache = os.path.join(CACHE_DIR, 'pca_variance_cache.npz')
    key = f'{os.path.getsize(EXPR)}_{int(mask_var.sum())}_{N_TOP_GENES}_{len(wanted)}'
    top_idx = None
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if str(z['key']) == key:
            top_idx = z['top_idx']
            var = z['var']
            log(f'variance scan: reused cached top-{N_TOP_GENES} selection')
    if top_idx is None:
        var = np.zeros(G, dtype=np.float64)
        for start in range(0, G, BATCH):
            stop = min(start + BATCH, G)
            use_cols = [gene_cols[i] for i in range(start, stop)]
            chunk = pd.read_csv(EXPR, usecols=use_cols, dtype=np.float32, engine='c',
                                low_memory=True)
            block = chunk.to_numpy(dtype=np.float64, copy=False)[mask_var]
            del chunk
            with np.errstate(invalid='ignore'):
                var[start:stop] = np.nanvar(block, axis=0, ddof=1)
            log(f'  variance {stop}/{G}')
        var = np.where(np.isnan(var), 0, var)
        top_idx = np.argsort(var)[::-1][:N_TOP_GENES]
        np.savez(cache, key=key, top_idx=top_idx, var=var)
        log('variance scan: selection cached')
    top_genes = [genes[i] for i in top_idx]
    log(f'top-{N_TOP_GENES} variance genes selected '
        f'({var[top_idx[0]]:.3f} … {var[top_idx[-1]]:.3f})')

    file_cols = sorted(gene_cols[i] for i in top_idx)
    pos_of = {fc: k for k, fc in enumerate(file_cols)}
    order_back = [pos_of[gene_cols[i]] for i in top_idx]

    mat = pd.read_csv(EXPR, usecols=file_cols, dtype=np.float32,
                      engine='c').to_numpy(dtype=np.float32)
    X_fit = mat[mask_var][:, order_back]
    X_all = mat[mask_all][:, order_back]
    ids_all = file_ids[mask_all]
    del mat

    miss = np.isnan(X_fit).mean(axis=0)
    keep = miss <= 0.2
    X_fit, X_all = X_fit[:, keep], X_all[:, keep]
    kept_genes = [g for g, k in zip(top_genes, keep) if k]
    med = np.nanmedian(X_fit, axis=0)
    X_fit = np.where(np.isnan(X_fit), med, X_fit).astype(np.float32)
    X_all = np.where(np.isnan(X_all), med, X_all).astype(np.float32)
    log(f'genes kept (<=20% missing): {int(keep.sum())}/{len(keep)}')

    n_comp = int(min(128, X_fit.shape[0], X_fit.shape[1]))
    scaler = StandardScaler().fit(X_fit)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(scaler.transform(X_fit))
    P = pca.transform(scaler.transform(X_all))
    comp = pca.components_                      # (n_comp, n_genes_kept)
    log(f'PCA-{n_comp} explained variance: {pca.explained_variance_ratio_.sum():.4f}')

    # ---- verify against the shipped artifact BEFORE writing ----
    shipped = pd.read_parquet(SHIPPED)
    if shipped.shape[0] != P.shape[0] or \
            [c for c in shipped.columns if c.startswith('pca_')] != \
            [f'pca_{i}' for i in range(P.shape[1])]:
        sys.exit(f'layout mismatch: shipped {shipped.shape} vs refit {P.shape}')
    if list(shipped['ModelID'].astype(str)) != list(ids_all):
        sys.exit('row order differs from the shipped parquet; aborting')
    ref = shipped[[f'pca_{i}' for i in range(P.shape[1])]].to_numpy()
    maxdiff = float(np.abs(ref - P).max())
    corr = float(np.corrcoef(ref.ravel(), P.ravel())[0, 1])

    # Subspace test: if the gene set matches, the shipped scores must lie in the
    # row space of this refit (residual ~0) even when components differ by sign or
    # order. A large residual means the gene set itself is different -> no gene
    # attribution is safe to publish.
    Q, *_ = np.linalg.lstsq(P, ref, rcond=None)
    residual = float(np.abs(ref - P @ Q).max())
    per_comp = [float(np.corrcoef(P[:, i], ref[:, i])[0, 1]) for i in range(P.shape[1])]
    aligned = sum(1 for c in per_comp if abs(c) > 0.999)
    np.savez(os.path.join(CACHE_DIR, 'pca_refit_debug.npz'),
             P=P, ref=ref, components=comp, ids=ids_all,
             kept_genes=np.array(kept_genes),
             explained=pca.explained_variance_ratio_)
    log(f'verification vs shipped parquet: max|diff|={maxdiff:.2e} corr={corr:.8f}')
    log(f'subspace test: max residual={residual:.3e} | components matching |corr|>0.999: '
        f'{aligned}/{P.shape[1]}')
    if not (maxdiff < TOL or corr > 0.999):
        if residual < 1e-3 and aligned >= P.shape[1] - 2:
            log('scores agree up to component sign/order — aligning before export')
            order = {}
            for i, c in enumerate(per_comp):
                order[i] = int(np.argmax([abs(np.corrcoef(P[:, j], ref[:, i])[0, 1])
                                          for j in range(P.shape[1])]))
            if len(set(order.values())) == P.shape[1]:
                comp = comp[[order[i] for i in range(P.shape[1])]]
                log('components realigned to the shipped column order')
            else:
                sys.exit('component mapping is not a permutation; aborting')
        else:
            sys.exit(f'refit does not reproduce the shipped PCA (maxdiff={maxdiff:.2e}, '
                     f'corr={corr:.8f}, residual={residual:.2e}); loadings would be wrong '
                     f'— aborting')

    # ---- export loadings ----
    rows = []
    for i in range(comp.shape[0]):
        w = comp[i]
        top = np.argsort(np.abs(w))[::-1][:3]
        rows.append({
            'pca_feature': f'pca_{i}',
            'explained_variance_ratio': round(float(pca.explained_variance_ratio_[i]), 6),
            'gene_1': kept_genes[top[0]], 'loading_1': round(float(w[top[0]]), 4),
            'gene_2': kept_genes[top[1]], 'loading_2': round(float(w[top[1]]), 4),
            'gene_3': kept_genes[top[2]], 'loading_3': round(float(w[top[2]]), 4),
        })
    ldf = pd.DataFrame(rows)
    ldf['label'] = ldf.apply(lambda r: f"{r['pca_feature']} "
                                   f"({r['gene_1']}, {r['gene_2']}, {r['gene_3']})", axis=1)
    ldf.to_csv(os.path.join(OUT, 'pca_loadings_top.csv'), index=False)
    np.save(os.path.join(OUT, 'pca_components.npy'), comp)
    pd.Series(kept_genes).to_csv(os.path.join(OUT, 'pca_kept_genes.csv'),
                                  index=False, header=['gene'])
    json.dump({'n_components': int(n_comp), 'n_genes_kept': int(len(kept_genes)),
               'explained_variance_sum': float(pca.explained_variance_ratio_.sum()),
               'verified_max_abs_diff_vs_shipped': maxdiff,
               'verified_correlation_vs_shipped': corr,
               'holdout_lines': sorted(holdout), 'fit_rows': int(mask_var.sum()),
               'dataset_rows': int(mask_all.sum())},
              open(os.path.join(OUT, 'pca_loadings_meta.json'), 'w'), indent=2)
    log(f'saved pca_loadings_top.csv ({len(ldf)} PCs), pca_components.npy')
    log('example labels:')
    for r in ldf.head(5).to_dict('records'):
        log(f"  {r['pca_feature']} -> {r['label']}")


if __name__ == '__main__':
    main()