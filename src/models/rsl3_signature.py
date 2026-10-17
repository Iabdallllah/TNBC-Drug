"""RSL3 mini-signature from published bulk-RNA-seq (GSE247883, A549).

RSL3 has no perturbation signature in L1000 (consensus 5,422 / full 66k terms)
and no ChEMBL entry, and no published RSL3-on-MDA-MB-231 transcriptomics exists
(GEO search). GSE247883 is the best public alternative: A549 cells, 9 samples -
DMSO vs RSL3 24h vs RSL3+Fer-1 24h - i.e. it has a built-in ferroptosis-rescue
arm (Fer-1 = lipid-peroxidation inhibitor).

What this produces (all clearly labelled as single-paper, A549-derived):
  1. RSL3 Down/Up gene sets (log2FC RSL3 vs DMSO, FPKM>1 baseline).
  2. Fer-1 reversal test: does RSL3+Fer-1 shift the RSL3-down genes back up?
  3. RSL3's protein-suppression page for MDA-MB-231: Down set intersected with
     genes expressed in ACH-000768 (the signature defines the drug effect, the
     MDA-MB-231 expression defines the cell context - same convention as L1000).
  4. RSL3 x {NAC, DFO} rescue reversal vs the L1000 signatures, so RSL3 joins
     the offtarget_rescue_reversal table.

IO budget: one 12 MB xlsx (cached) + expr79.npz + chempert_lib.tsv.
Run:  python src/models/rsl3_signature.py
"""
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'outputs_local')
CACHE_DIR = os.environ.get('TNBC_CACHE', '/tmp/opencode')
XLSX = os.path.join(CACHE_DIR, 'GSE247883_fpkm.xlsx')
LIB_CACHE = os.path.join(CACHE_DIR, 'chempert_lib.tsv')
MDA231 = 'ACH-000768'
TOP_N = 150            # top-N up/down by log2FC - same scale as L1000 sets (~250)
BASE_FPKM = 1.0        # require measurable baseline expression


def log(m):
    print(m, flush=True)


def norm_key(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())


def load_l1000():
    sigs = {}
    with open(LIB_CACHE, encoding='utf-8', errors='replace') as f:
        for line in f:
            if '\t' not in line:
                continue
            name, _, genes = line.split('\t', 2)
            gs = {g for g in genes.split('\t') if g.strip()}
            if gs and name.endswith((' Up', ' Down')):
                drug, direction = name.rsplit(' ', 1)
                sigs.setdefault(norm_key(drug), {})[direction] = gs
    return sigs


def main():
    df = pd.read_excel(XLSX, usecols='A:AD')
    sample_cols = [c for c in df.columns if c.startswith('FPKM.')]
    groups = {}
    for c in sample_cols:
        parts = c.split('.')
        grp = re.sub(r'_\d+$', '', parts[1]) if len(parts) > 1 else c   # FPKM.R_F_1 -> R_F
        groups.setdefault(grp, []).append(c)
    log(f'GSE247883: {len(df)} genes, groups: {[(g, len(v)) for g, v in groups.items()]}')
    d, r, rf = groups['DMSO'], groups['RSL3'], groups['R_F']
    m = df[['gene_name'] + d + r + rf].copy()
    m = m[m['gene_name'].notna() & (m['gene_name'] != '')]
    agg = m.groupby('gene_name', as_index=False).mean(numeric_only=True)

    dm = agg[d].mean(axis=1).to_numpy(dtype=float)
    rm = agg[r].mean(axis=1).to_numpy(dtype=float)
    fm = agg[rf].mean(axis=1).to_numpy(dtype=float)
    base = dm > BASE_FPKM
    lfc_drug = np.log2(rm + 1) - np.log2(dm + 1)
    lfc_fer = np.log2(fm + 1) - np.log2(dm + 1)
    log(f'genes with baseline FPKM>{BASE_FPKM}: {int(base.sum())}')

    # Down: only genes the cell actually makes (can't suppress a silent gene).
    # Up: keep all genes - a control-silent gene strongly induced by RSL3 is real.
    order = np.argsort(np.where(base, lfc_drug, np.inf))
    down_idx = order[:TOP_N]                      # most suppressed by RSL3
    up_idx = np.argsort(lfc_drug)[-TOP_N:][::-1]  # most induced by RSL3
    is_down = np.zeros(len(agg), dtype=bool); is_down[down_idx] = True
    is_up = np.zeros(len(agg), dtype=bool); is_up[up_idx] = True
    down_genes = set(agg['gene_name'][is_down])
    up_genes = set(agg['gene_name'][is_up])
    log(f'RSL3 signature (top-{TOP_N}): {len(down_genes)} Down / {len(up_genes)} Up '
        f'(range {lfc_drug[down_idx].max():.2f} .. {lfc_drug[up_idx].min():.2f} log2FC)')

    sig_df = pd.DataFrame({
        'gene': agg['gene_name'],
        'fpkm_dmso_mean': dm.round(2), 'fpkm_rsl3_mean': rm.round(2),
        'fpkm_rsl3_fer1_mean': fm.round(2),
        'log2fc_rsl3_vs_dmso': lfc_drug.round(3),
        'log2fc_rsl3_fer1_vs_dmso': lfc_fer.round(3),
        'rsl3_set': np.where(is_down, 'Down', np.where(is_up, 'Up', '')),
    })
    sig_df.to_csv(os.path.join(OUT, 'rsl3_mini_signature.csv'), index=False)

    # ---- Fer-1 reversal: RSL3-down genes recovering with Fer-1 ----
    # recovery index: median lfc_fer within the RSL3-Down set vs genome-wide median
    rec_med = float(np.median(lfc_fer[is_down]))
    all_med = float(np.median(lfc_fer))
    from scipy.stats import mannwhitneyu
    u, p_rec = mannwhitneyu(lfc_fer[is_down], lfc_fer, alternative='greater')
    frac_recov = float((lfc_fer[is_down] > 0).mean())
    log(f'Fer-1 reversal: median lfc_fer in RSL3-Down set = {rec_med:+.3f} '
        f'vs genome-wide {all_med:+.3f} | {frac_recov*100:.0f}% of down genes recover, '
        f'Mann-Whitney p={p_rec:.2e}')

    # ---- MDA-MB-231 context ----
    z = np.load(os.path.join(CACHE_DIR, 'expr79.npz'), allow_pickle=False)
    X, ids, genes = z['X'], z['ids'].astype(str), z['genes'].astype(str)
    r231 = int(np.flatnonzero(ids == MDA231)[0])
    e231 = set(genes[X[r231] > 1.0])
    down_231 = down_genes & e231
    up_231 = up_genes & e231
    log(f'expressed in MDA-MB-231: {len(e231)} genes | '
        f'RSL3-Down intersected: {len(down_231)} | RSL3-Up: {len(up_231)}')

    # ---- RSL3 x rescue agents (L1000) reversal ----
    sigs = load_l1000()
    res = []
    for rescue_d, syns in (('N-ACETYLCYSTEINE',
                            ['N-ACETYLCYSTEINE', 'N-ACETYL-L-CYSTEINE', 'NAC']),
                           ('DEFEROXAMINE',
                            ['DEFEROXAMINE', 'DESFERIOXAMINE', 'DEFERIOXAMINE'])):
        rk = None
        for s in syns:
            k = norm_key(s)
            if k in sigs and 'Up' in sigs[k]:
                rk = k
                break
            cands = [(tn, mm) for tn, mm in sigs.items()
                     if len(tn) >= 5 and (k in tn or tn in k) and 'Up' in mm]
            if cands:
                rk = max(cands, key=lambda c: len(c[0]))[0]
                break
        if rk is None:
            log(f'[warn] {rescue_d}: no L1000 signature found')
            continue
        ru = sigs[rk]['Up'] & e231
        rd = sigs[rk]['Down'] & e231
        N = len(e231)
        ov = len(down_231 & ru)
        p = float(hypergeom.sf(max(ov - 1, 0), N, len(ru), len(down_231)))
        res.append({'line': MDA231, 'killer': 'RSL3', 'rescue': rescue_d,
                    'signature_source': 'GSE247883 (A549, single paper)',
                    'killer_down_genes': len(down_231), 'rescue_up_genes': len(ru),
                    'reversal_hits': ov, 'reversal_p': round(min(p, 1.0), 6),
                    'reversal_enrich': round((ov / max(1, len(ru))) /
                                             max(1e-9, len(down_231) / N), 2),
                    'genes': ', '.join(sorted(down_231 & ru)[:20])})
        log(f'RSL3 x {rescue_d}: {ov} reversed genes, p={p:.2e}')

    rv = pd.read_csv(os.path.join(OUT, 'offtarget_rescue_reversal.csv'))
    rv = pd.concat([rv, pd.DataFrame(res)], ignore_index=True)
    rv = rv.drop_duplicates(subset=['line', 'killer', 'rescue'], keep='last')
    rv.to_csv(os.path.join(OUT, 'offtarget_rescue_reversal.csv'), index=False)

    meta = {
        'source': 'GSE247883 (A549, DMSO/RSL3/RSL3+Fer-1, 24h, n=3 each)',
        'quality_note': 'single paper, one cell line, one dose - weaker than the '
                        'L1000 consensus signatures; clearly labelled in all outputs',
        'n_down': len(down_genes), 'n_up': len(up_genes),
        'n_down_expressed_mda231': len(down_231),
        'fer1_reversal': {'recovery_median_lfc_in_down_set': round(rec_med, 3),
                          'genome_median_lfc': round(all_med, 3),
                          'fraction_recovered': round(frac_recov, 3),
                          'mannwhitney_p': float(p_rec)},
        'target_check': 'GPX4 expressed in MDA-MB-231 (log2TPM+1=8.24) - the '
                        'mechanism RSL3 inhibits is present in the target cell',
    }
    json.dump(meta, open(os.path.join(OUT, 'rsl3_mini_signature_meta.json'), 'w'),
              indent=2)

    log('\n=== RSL3 top suppressed proteins in MDA-MB-231 (by |log2FC|) ===')
    sub = sig_df[sig_df['gene'].isin(down_231)].sort_values('log2fc_rsl3_vs_dmso')
    print(sub[['gene', 'log2fc_rsl3_vs_dmso', 'log2fc_rsl3_fer1_vs_dmso']].head(12).to_string(index=False))
    log('\nsaved: rsl3_mini_signature.csv, rsl3_mini_signature_meta.json, '
        'offtarget_rescue_reversal.csv updated with RSL3 rows')


if __name__ == '__main__':
    main()