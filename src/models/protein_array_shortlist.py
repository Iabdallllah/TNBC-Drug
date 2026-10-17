"""Protein-array shortlist for the FerroSelect wet-lab panel.

Ranks candidate proteins per panel member (single agents + killer x statin
combos, additive union of Down signatures) for measurement on a protein array
in MDA-MB-231.

Priority = rescue-competent (gene sits in the killer-vs-rescue reversal overlap,
i.e. NAC/DFO statistically reverse it) > membrane/secreted (easiest on an array,
from UniProt subcellular annotations, one batch query, cached) > effect size.

IO budget: caches only + one UniProt batch call.
Run:  python src/models/protein_array_shortlist.py
"""
import json
import os
import re
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'outputs_local')
CACHE_DIR = os.environ.get('TNBC_CACHE', '/tmp/opencode')
MDA231 = 'ACH-000768'
STATIN = 'SIMVASTATIN'          # the panel statin (lovastatin run separately)

COMBOS = [('ERASTIN+' + STATIN, 'ERASTIN', STATIN),
          ('SULFASALAZINE+' + STATIN, 'SULFASALAZINE', STATIN),
          ('RSL3+' + STATIN, 'RSL3', STATIN)]


def log(m):
    print(m, flush=True)


def norm_key(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())


def load_l1000():
    sigs = {}
    with open(os.path.join(CACHE_DIR, 'chempert_lib.tsv'),
              encoding='utf-8', errors='replace') as f:
        for line in f:
            if '\t' not in line:
                continue
            name, _, genes = line.split('\t', 2)
            gs = {g for g in genes.split('\t') if g.strip()}
            if gs and name.endswith((' Up', ' Down')):
                drug, direction = name.rsplit(' ', 1)
                sigs.setdefault(norm_key(drug), {})[direction] = gs
    return sigs


def resolve(sigs, name, syns):
    for s in syns:
        k = norm_key(s)
        if k in sigs and 'Down' in sigs[k]:
            return sigs[k]
        cands = [(tn, m) for tn, m in sigs.items()
                 if len(tn) >= 5 and (k in tn or tn in k) and 'Down' in m]
        if cands:
            return max(cands, key=lambda c: len(c[0]))[1]
    return None


def uniprot_localization(gene_list, cache_key):
    """gene -> 'membrane' / 'secreted' / '' via batch UniProt queries (cached)."""
    cache = os.path.join(CACHE_DIR, cache_key)
    if os.path.exists(cache):
        return json.load(open(cache))
    out = {}
    for i in range(0, len(gene_list), 100):          # URL-length-safe batches
        batch = gene_list[i:i + 100]
        q = ' OR '.join(f'gene_exact:{g}' for g in batch)
        url = ('https://rest.uniprot.org/uniprotkb/search?' +
               urllib.parse.urlencode({
                   'query': f'({q}) AND organism_id:9606 AND reviewed:true',
                   'fields': 'gene_primary,cc_subcellular_location', 'format': 'tsv',
                   'size': '500'}))
        req = urllib.request.Request(url, headers={'User-Agent': 'tnbc-thesis/1.0'})
        rows = urllib.request.urlopen(req, timeout=90).read().decode('utf-8', 'replace')
        for line in rows.splitlines()[1:]:
            parts = line.split('\t')
            gene, loc = (parts + ['', ''])[:2]
            tag = ''
            low = loc.lower()
            if 'secret' in low or 'extracellular' in low:
                tag = 'secreted'
            elif 'membrane' in low:
                tag = 'membrane'
            out[gene] = tag
    json.dump(out, open(cache, 'w'))
    return out


def main():
    z = np.load(os.path.join(CACHE_DIR, 'expr79.npz'), allow_pickle=False)
    X, ids, genes = z['X'], z['ids'].astype(str), z['genes'].astype(str)
    r = int(np.flatnonzero(ids == MDA231)[0])
    e231 = set(genes[X[r] > 1.0])
    log(f'MDA-MB-231 expressed genes: {len(e231)}')

    sigs = load_l1000()
    # RSL3 from the GSE247883 mini-signature (single paper, labelled)
    rsl3_df = pd.read_csv(os.path.join(OUT, 'rsl3_mini_signature.csv'))
    down_sets = {
        'ERASTIN': resolve(sigs, 'ERASTIN', ['ERASTIN'])['Down'] & e231,
        'SULFASALAZINE': resolve(sigs, 'SULFASALAZINE', ['SULFASALAZINE'])['Down'] & e231,
        STATIN: resolve(sigs, STATIN, [STATIN])['Down'] & e231,
        'RSL3': set(rsl3_df.loc[rsl3_df.rsl3_set == 'Down', 'gene']) & e231,
    }

    # rescue-reversal overlaps (genes NAC/DFO restore) from the reversal table
    rv = pd.read_csv(os.path.join(OUT, 'offtarget_rescue_reversal.csv'))
    reversal_genes = {}
    for _, row in rv[rv.line == MDA231].iterrows():
        reversal_genes.setdefault(row['killer'], set()).update(
            str(row['genes']).split(', ') if isinstance(row['genes'], str) else [])

    # combos: additive union of Down sets (+ shared core flagged)
    for label, k1, k2 in COMBOS:
        down_sets[label] = down_sets[k1] | down_sets[k2]

    rows = []
    for drug, down in down_sets.items():
        loc = uniprot_localization(sorted(down), 'uniprot_shortlist.json')
        for g in sorted(down):
            rows.append({
                'drug': drug, 'gene': g,
                'in_rescue_reversal': int(g in reversal_genes.get(drug.split('+')[0], set())),
                'array_friendly': loc.get(g, ''),
                'in_statin_core': int(drug.endswith('+' + STATIN) and
                                      g in down_sets[drug.split('+')[0]] and
                                      g in down_sets[STATIN]),
            })
    short = pd.DataFrame(rows)
    short = short.sort_values(['drug', 'in_rescue_reversal', 'array_friendly',
                               'in_statin_core'],
                              ascending=[True, False, False, False])
    short.to_csv(os.path.join(OUT, 'protein_array_shortlist.csv'), index=False)

    log('\n=== shortlist preview (top 5 per drug) ===')
    for drug in short['drug'].unique():
        sub = short[short.drug == drug].head(5)
        print(f'{drug:30} ' + ', '.join(sub['gene']))
    n_resc = int(short.in_rescue_reversal.sum())
    n_arr = int((short.array_friendly != '').sum())
    log(f'\nsaved protein_array_shortlist.csv | {len(short)} candidates | '
        f'{n_resc} rescue-competent | {n_arr} membrane/secreted')


if __name__ == '__main__':
    main()