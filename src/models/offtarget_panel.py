"""Off-target protein-effect layer for the FerroSelect wet-lab panel.

Question it answers
-------------------
The panel drugs (erastin, sulfasalazine +/- statin, with NAC/deferoxamine as
rescue controls, RSL3 lab-only) kill the cancer cell. Do they also suppress the
expression of OTHER proteins in that cell - the side-effect profile - and can
the rescue logic (NAC/DFO reversing the killer's effects) be shown in silico?

Method
------
1. L1000 chem-pert consensus signatures (drug Up/Down gene sets) from the cached
   Enrichr library. RSL3 has no signature anywhere (L1000/ChEMBL) -> lab-only.
2. Intersect each signature with the genes actually expressed in the target
   line (log2(TPM+1) > 1), so we only list proteins the cell can make.
3. Rescue test: a rescue agent should UP-regulate genes the killer DOWN-regulates
   (and vice versa). Scored with hypergeometric enrichment of that reciprocal
   overlap among expressed genes.

IO budget: cache file only (expr79.npz ~25 MB + chempert_lib.tsv 16 MB) - no
full-CSV reads, no model runs.
Run:  python src/models/offtarget_panel.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'outputs_local')
CACHE_DIR = os.environ.get('TNBC_CACHE', '/tmp/opencode')
EXPR_CACHE = os.path.join(CACHE_DIR, 'expr79.npz')
LIB_CACHE = os.path.join(CACHE_DIR, 'chempert_lib.tsv')
EXPR_THRESHOLD = 1.0          # log2(TPM+1) > 1  <=>  TPM > 1
LINES = ['ACH-000768', 'ACH-000849', 'ACH-000019']   # 231, 468, MCF7
KILLERS = ['ERASTIN', 'SULFASALAZINE', 'SIMVASTATIN', 'LOVASTATIN']
RESCUE = ['N-ACETYLCYSTEINE', 'DEFEROXAMINE']
SYNONYMS = {                       # library term spellings differ from PRISM names
    'ERASTIN': ['ERASTIN'],
    'SULFASALAZINE': ['SULFASALAZINE'],
    'N-ACETYLCYSTEINE': ['N-ACETYLCYSTEINE', 'N-ACETYL-L-CYSTEINE', 'NAC'],
    'DEFEROXAMINE': ['DEFEROXAMINE', 'DESFERIOXAMINE', 'DEFERIOXAMINE',
                     'DESFERIOXAMINE-MESYLATE'],
    'SIMVASTATIN': ['SIMVASTATIN'],
    'LOVASTATIN': ['LOVASTATIN'],
}


def log(m):
    print(m, flush=True)


def norm_key(s):
    import re
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())


def load_signatures():
    """drug key -> {'Up': set(genes), 'Down': set(genes)} from the cached library."""
    sigs = {}
    with open(LIB_CACHE, encoding='utf-8', errors='replace') as f:
        for line in f:
            if '\t' not in line:
                continue
            name, _, genes = line.split('\t', 2)
            gs = {g for g in genes.split('\t') if g.strip()}
            if not gs or not name.endswith((' Up', ' Down')):
                continue
            drug, direction = name.rsplit(' ', 1)
            sigs.setdefault(norm_key(drug), {})[direction] = gs
    log(f'signature library: {len(sigs)} drugs with Up/Down sets')
    return sigs


def main():
    z = np.load(EXPR_CACHE, allow_pickle=False)
    X, ids, genes = z['X'], z['ids'].astype(str), z['genes'].astype(str)
    sigs = load_signatures()

    drugs = KILLERS + RESCUE
    drug_keys = {}
    for d in drugs:
        for s in SYNONYMS.get(d, [d]):
            k = norm_key(s)
            if k in sigs and 'Up' in sigs[k] and 'Down' in sigs[k]:
                drug_keys[d] = k
                break
            cands = [(tn, m) for tn, m in sigs.items()
                     if len(tn) >= 5 and (k in tn or tn in k)
                     and 'Up' in m and 'Down' in m]
            if cands:
                drug_keys[d] = max(cands, key=lambda c: len(c[0]))[0]
                break
        if d not in drug_keys:
            log(f'[warn] {d}: no usable signature - skipped')

    # ---- per-drug, per-line expressed intersection ----
    rows = []
    for d, k in drug_keys.items():
        for lid in LINES:
            r = int(np.flatnonzero(ids == lid)[0])
            expr = X[r] > EXPR_THRESHOLD
            egenes = set(genes[expr])
            for direction in ('Down', 'Up'):
                inter = sigs[k][direction] & egenes
                rows.append({
                    'drug': d, 'direction': direction, 'line': lid,
                    'sig_genes': len(sigs[k][direction]),
                    'expressed_genes': len(egenes),
                    'hit_genes': len(inter),
                    'hit_fraction': round(len(inter) / max(1, len(sigs[k][direction])), 3),
                    'genes': ', '.join(sorted(inter)[:25]),
                })
    amap = pd.DataFrame(rows)
    amap.to_csv(os.path.join(OUT, 'offtarget_panel_map.csv'), index=False)

    # ---- rescue reversal test (reciprocal overlap within expressed genes) ----
    res = []
    for lid in LINES:
        r = int(np.flatnonzero(ids == lid)[0])
        egenes = set(genes[X[r] > EXPR_THRESHOLD])
        N = len(egenes)
        for killer in KILLERS:
            if killer not in drug_keys:
                continue
            k = drug_keys[killer]
            for rescue_d in RESCUE:
                if rescue_d not in drug_keys:
                    continue
                rk = drug_keys[rescue_d]
                kd, ru = sigs[k]['Down'] & egenes, sigs[rk]['Up'] & egenes
                ku, rd = sigs[k]['Up'] & egenes, sigs[rk]['Down'] & egenes
                ov1, ov2 = len(kd & ru), len(ku & rd)
                # enrichment of reciprocal overlap vs a random gene set of same size
                M = N                     # population: expressed genes
                n_res, n_kd = len(ru), len(kd)
                p1 = float(hypergeom.sf(max(ov1 - 1, 0), M, n_res, n_kd)) if n_res and n_kd else 1.0
                n_ru2, n_ku = len(rd), len(ku)
                p2 = float(hypergeom.sf(max(ov2 - 1, 0), M, n_ru2, n_ku)) if n_ru2 and n_ku else 1.0
                res.append({
                    'line': lid, 'killer': killer, 'rescue': rescue_d,
                    'killer_down_genes': n_kd, 'rescue_up_genes': n_res,
                    'reversal_hits': ov1, 'reversal_p': round(min(p1, p2), 6),
                    'reversal_enrich': round((ov1 / max(1, n_res)) / max(1e-9, n_kd / M), 2),
                    'genes': ', '.join(sorted(kd & ru)[:20]),
                })
    rv = pd.DataFrame(res)
    if rv.empty:
        sys.exit('no killer x rescue pair resolved - check signatures')
    rv.to_csv(os.path.join(OUT, 'offtarget_rescue_reversal.csv'), index=False)

    log('\n=== expressed-protein suppression (MDA-MB-231) ===')
    print(amap[(amap.line == 'ACH-000768')]
          .pivot_table(index='drug', columns='direction', values='hit_genes')
          .loc[list(drug_keys)].to_string())
    log('\n=== in-silico rescue reversal (best line per pair, min p) ===')
    best = rv.sort_values('reversal_p').groupby(['killer', 'rescue']).head(1)
    print(best[['killer', 'rescue', 'line', 'reversal_hits', 'reversal_p',
                'reversal_enrich']].to_string(index=False))
    log('\nsaved: offtarget_panel_map.csv, offtarget_rescue_reversal.csv')


if __name__ == '__main__':
    main()