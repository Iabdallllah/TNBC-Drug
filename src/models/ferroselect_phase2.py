"""FerroSelect track, phase 2 (proposal §6.1: small model + Aim 0).

Needs Data/OmicsExpressionTPMLogp1HumanAllGenes.csv (raw gene-level matrix).

- CORE set: 10 proposal-named ferroptosis genes (GPX4, SLC7A11, GCLC, GSS,
  ACSL4, LPCAT3, NFE2L2, FTH1, TFRC, SLC3A2). Flagged as the core-axis set;
  full-FerrDb enrichment is a follow-up when those tables land.
- Small model: 10 core genes + drug one-hot, Lasso/Ridge/RF, nested
  cell-line GroupKFold (outer 3; inner built-in CV for linear, mini-grid for RF).
- Aim 0: labels_circ = rank-based ssGSEA-lite score of the core set per line
  (computed from the SAME expression matrix that supplies features —
  circular on purpose), same models/splits as the real-label run.
  Expected: inflated performance. gap = circ - real is THE reported number.
- Ceiling check applies to REAL-label runs only. For Aim 0, a high score is
  the demonstration, not a bug.

Outputs: ferroselect_phase2.json
"""
import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, GridSearchCV

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, 'outputs_local')
EXPR = os.path.join(HERE, 'Data', 'OmicsExpressionTPMLogp1HumanAllGenes.csv')
CORE = ['GPX4', 'SLC7A11', 'GCLC', 'GSS', 'ACSL4', 'LPCAT3',
        'NFE2L2', 'FTH1', 'TFRC', 'SLC3A2']
PANEL = ['DPC-006173', 'DPC-005959', 'DPC-003889', 'DPC-002562']
SEED = 42
N_BG = 2000


def find_cols():
    with open(EXPR, newline='', encoding='utf-8', errors='replace') as f:
        hdr = next(csv.reader(f))
    low = [h.strip().lower() for h in hdr]
    id_pos = next(i for i, h in enumerate(low) if 'modelid' in h)
    core_pos = {}
    for g in CORE:
        hit = [i for i, h in enumerate(hdr)
               if h.startswith(g + ' ') or h.startswith(g + '(') or h == g]
        if hit:
            core_pos[g] = hit[0]
    gene_pos = [i for i, h in enumerate(hdr)
                if i != id_pos and '(' in h and ')' in h]
    return hdr, id_pos, core_pos, gene_pos


def load_block(usecols):
    return pd.read_csv(EXPR, usecols=sorted(set(usecols)), engine='c',
                       low_memory=True)


def ssgsea_lite(mat, set_idx):
    """Rank-based single-sample enrichment of columns set_idx within each row.
    mat: (n_samples, n_genes incl. background). Returns score per row."""
    r = np.argsort(np.argsort(-mat, axis=1), axis=1).astype(float)  # 0 = top
    n = mat.shape[1]
    m = len(set_idx)
    hit = np.zeros_like(r)
    hit[:, set_idx] = 1.0
    p_hit = (1.0 / m) * (hit * (n - r)).sum(axis=1) / n * n
    # classic walk: reward hits by top-rank weight, penalize misses uniformly
    order = np.argsort(r, axis=1)
    ranked_hit = np.take_along_axis(hit, order, axis=1)
    w = (n - np.arange(n)) / n
    walk = np.cumsum(ranked_hit * w - (1 - ranked_hit) / (n - m), axis=1)
    es = walk[np.arange(len(walk)), np.abs(walk).argmax(axis=1)]
    return es


def oof_eval(X, y, g, models, outer=3):
    gkf = GroupKFold(n_splits=outer)
    oof = {n: np.full(len(y), np.nan) for n in models}
    for tr, te in gkf.split(X, y, g):
        for name, mk in models.items():
            from sklearn.base import clone
            from sklearn.model_selection import GridSearchCV
            est = clone(mk)
            if isinstance(est, GridSearchCV):
                est.fit(X[tr], y[tr], groups=g[tr])
            else:
                est.fit(X[tr], y[tr])
            oof[name][te] = est.predict(X[te])
    res = {}
    for name in models:
        yt = np.asarray(y, dtype=float)
        yp = np.asarray(oof[name], dtype=float)
        try:
            sp = float(spearmanr(yt, yp).statistic)
        except Exception:
            sp = float('nan')
        res[name] = {'rmse': float(mean_squared_error(yt, yp) ** 0.5),
                     'spearman': sp}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=OUT)
    a = ap.parse_args()

    hdr, id_pos, core_pos, gene_pos = find_cols()
    missing = [g for g in CORE if g not in core_pos]
    print(f'core genes found: {len(core_pos)}/{len(CORE)}'
          + (f' MISSING: {missing}' if missing else ''))
    assert len(core_pos) >= 8, 'too many core genes missing from matrix'
    core = [g for g in CORE if g in core_pos]

    rng = np.random.RandomState(SEED)
    pool = [i for i in gene_pos if i not in set(core_pos.values())]
    bg = sorted(rng.choice(pool, size=min(N_BG, len(pool)), replace=False).tolist())
    print(f'background genes: {len(bg)}')

    ids = pd.read_csv(EXPR, usecols=[id_pos], engine='c', low_memory=True)
    id_col = ids.columns[0]
    ml = pd.read_parquet(os.path.join(a.outdir, 'tnbc_ml_dataset.parquet'),
                         columns=['ModelID', 'drug_id', 'auc'])
    ml = ml.drop_duplicates(subset=['ModelID', 'drug_id']).reset_index(drop=True)
    panel = ml[ml['drug_id'].isin(PANEL)].reset_index(drop=True)
    lines = sorted(panel['ModelID'].unique())
    print(f'panel rows: {len(panel)} | lines: {len(lines)}')

    # expression for panel lines: core + background
    want_cols = sorted({id_pos} | {core_pos[g] for g in core} | set(bg))
    emo = load_block(want_cols)
    emap = {orig: k for k, orig in enumerate(sorted({id_pos} | {core_pos[g] for g in core} | set(bg)))}
    core_idx = [emap[core_pos[g]] for g in core]
    bg_idx = [emap[i] for i in bg]
    emo = emo.drop_duplicates(subset=[emo.columns[0]], keep='first').reset_index(drop=True)
    emo = emo[emo[emo.columns[0]].astype(str).isin(lines)].reset_index(drop=True)
    got = emo[emo.columns[0]].astype(str).tolist()
    print(f'panel lines with expression: {len(got)}/{len(lines)}')
    # E columns == order_all[1:] (order_all[0] is the ID column)
    order_all = sorted({id_pos} | {core_pos[g] for g in core} | set(bg))
    ecols = [c for c in emo.columns[1:]]
    E = emo[ecols].to_numpy(dtype=np.float64)
    idv = emo[emo.columns[0]].astype(str).to_numpy()
    set_idx = [order_all.index(core_pos[g]) - 1 for g in core]
    assert all(0 <= j < E.shape[1] for j in set_idx)
    scores = ssgsea_lite(E, set_idx)
    smap = dict(zip(idv, scores))
    print('ssGSEA-lite range:', round(float(scores.min()), 3), '..', round(float(scores.max()), 3))

    # design matrices (core genes + drug one-hot)
    ecol_pos = {c: j for j, c in enumerate(ecols)}
    # map each core gene -> its column in E via file position
    core_eidx = {}
    for g in core:
        fp = core_pos[g]
        colname = emo.columns[1:][order_all.index(fp) - 1]
        core_eidx[g] = ecol_pos[colname]
    drugs = sorted(panel['drug_id'].unique())
    id2row = {v: i for i, v in enumerate(idv.tolist())}
    rows = []
    for _, r in panel.iterrows():
        lid = r['ModelID']
        if lid not in smap:
            continue
        feat = [E[id2row[lid], core_eidx[g]] for g in core]
        oh = [1.0 if d == r['drug_id'] else 0.0 for d in drugs]
        rows.append((lid, r['drug_id'], r['auc'], smap[lid], feat + oh))
    lab = np.array([x[3] for x in rows])   # circular labels
    real = np.array([x[2] for x in rows], dtype=float)
    X = np.array([x[4] for x in rows], dtype=np.float32)
    g = np.array([x[0] for x in rows])
    print(f'model rows: {len(rows)} (dropped {len(panel) - len(rows)} without expression)')

    base_models = {
        'Lasso': LassoCV(n_alphas=50, cv=3, random_state=SEED, max_iter=20000),
        'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 13), cv=3),
        'RandomForest': GridSearchCV(
            RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED),
            {'max_depth': [6, None], 'min_samples_leaf': [2, 5]},
            cv=GroupKFold(n_splits=2), n_jobs=1),
    }
    print('== REAL labels (measured AUC) ==')
    real_res = oof_eval(X, real, g, base_models)
    for n, m in real_res.items():
        print(f"  {n:12s} RMSE {m['rmse']:.4f} | Spearman {m['spearman']:.4f}")
    print('== AIM 0 labels (ssGSEA circular) ==')
    circ_res = oof_eval(X, lab, g, base_models)
    for n, m in circ_res.items():
        print(f"  {n:12s} RMSE {m['rmse']:.4f} | Spearman {m['spearman']:.4f}")
    gap = {n: (circ_res[n]['spearman'] - real_res[n]['spearman']
                if not (np.isnan(circ_res[n]['spearman']) or np.isnan(real_res[n]['spearman']))
                else None) for n in base_models}
    print('inflation gap (circ - real):',
          {n: (round(v, 4) if v is not None else None) for n, v in gap.items()})

    out = {'core_genes_used': core, 'core_missing': missing,
           'n_background': len(bg), 'n_rows': int(len(rows)),
           'n_lines': int(len(set(g))), 'real_labels': real_res,
           'aim0_labels': circ_res, 'inflation_gap': gap,
           'note': 'Aim-0 high scores demonstrate inflation, not a bug. '
                   'Core set = proposal axis (10 genes); full-FerrDb rerun pending tables.'}
    json.dump(out, open(os.path.join(a.outdir, 'ferroselect_phase2.json'), 'w'), indent=1)
    print('saved ferroselect_phase2.json')


if __name__ == '__main__':
    main()
