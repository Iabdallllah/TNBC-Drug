"""Coverage audit: are our top predicted killers covered by perturbation databases?

For each top-20 MDA-MB-231 predicted killer this checks:
  1. LINCS L1000 chem perturbation consensus signatures (Enrichr library,
     'drug Up' / 'drug Down' gene sets) - the data source for predicting which
     proteins a drug up/down-regulates.
  2. ChEMBL known mechanisms - the deterministic part of off-targets.

Network budget: one cached library download (~16 MB, /tmp/opencode) + 2 small
API calls per drug.
Run:  python src/models/offtarget_coverage.py
"""
import json
import os
import re
import urllib.parse
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'outputs_local')
CACHE_DIR = os.environ.get('TNBC_CACHE', '/tmp/opencode')
LIB_CACHE = os.path.join(CACHE_DIR, 'chempert_lib.tsv')
TOP20 = os.path.join(OUT, 'mda_mb_231_top20_killers.csv')
CHEMBL = 'https://www.ebi.ac.uk/chembl/api/data'
ENRICHR = 'https://maayanlab.cloud/Enrichr'
LIBRARY = 'LINCS_L1000_Chem_Pert_Consensus_Sigs'


def log(m):
    print(m, flush=True)


def http_get(url, timeout=45):
    req = urllib.request.Request(url, headers={'User-Agent': 'tnbc-thesis/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def norm(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())


def load_l1000_terms():
    """term -> n_genes, from the consensus chem-pert library (cached download)."""
    if not os.path.exists(LIB_CACHE):
        os.makedirs(CACHE_DIR, exist_ok=True)
        url = f'{ENRICHR}/geneSetLibrary?mode=text&libraryName={LIBRARY}'
        log(f'downloading {LIBRARY} ...')
        data = http_get(url, timeout=120)
        open(LIB_CACHE, 'wb').write(data)
    terms = {}
    with open(LIB_CACHE, encoding='utf-8', errors='replace') as f:
        for line in f:
            if '\t' not in line:
                continue
            name, _, genes = line.split('\t', 2)
            genes = [g for g in genes.split('\t') if g.strip()]
            if name.endswith((' Up', ' Down')) and genes:
                terms[norm(name.rsplit(' ', 1)[0])] = {
                    'term': name.strip(), 'n_genes': len(genes)}
    log(f'L1000 consensus library: {len(terms)} drug signatures')
    return terms


TARGET_CACHE = {}


def target_name(tid):
    """ChEMBL target -> (pref_name, target_type), cached per session."""
    if tid not in TARGET_CACHE:
        try:
            t = json.loads(http_get(f'{CHEMBL}/target/{tid}.json'))
            m = t.get('target', t) if isinstance(t, dict) else {}
            pref = m.get('pref_name') or ''
            pref = pref.split()[0] if pref.startswith('ChEBI:') else pref
            TARGET_CACHE[tid] = (pref or tid, m.get('target_type', ''))
        except Exception:
            TARGET_CACHE[tid] = (tid, '')
    return TARGET_CACHE[tid]


def chembl_lookup(name):
    """(chembl_id, max_phase, [mechanism texts]) - with a salt-stripped retry."""
    attempts = [name]
    stripped = re.sub(r'(?i)[- ]?(MESYLATE|HYDROCHLORIDE|\(\+\)-|\(-\)-|SODIUM)$', '',
                      name).strip()
    if norm(stripped) != norm(name):
        attempts.append(stripped)
    for q in attempts:
        try:
            res = json.loads(http_get(
                f'{CHEMBL}/molecule/search.json?q={urllib.parse.quote(q)}&limit=10'))
        except Exception as e:
            return None, None, [f'error: {e}']
        key = norm(q)
        hit = None
        for m in res.get('molecules', []):
            pn = norm(m.get('pref_name') or '')
            if pn == key or (len(key) >= 6 and key in pn):
                hit = m
                break
        if hit is None:
            continue
        cid, phase = hit.get('molecule_chembl_id'), hit.get('max_phase')
        mech, targets = [], []
        try:
            moa = json.loads(http_get(
                f'{CHEMBL}/mechanism.json?molecule_chembl_id={cid}&limit=10'))
            mech = [m.get('mechanism_of_action') or ''
                    for m in moa.get('mechanisms', []) if m.get('mechanism_of_action')]
        except Exception:
            pass
        if not mech:
            try:      # no curated MOA -> take its measured activity targets
                act = json.loads(http_get(
                    f'{CHEMBL}/activity.json?molecule_chembl_id={cid}'
                    f'&target_chembl_id__is_not_null&limit=15'))
                tids = list(dict.fromkeys(
                    a.get('target_chembl_id') for a in act.get('activities', [])))[:5]
                targets = [f'{target_name(t)[0]}' for t in tids]
            except Exception:
                pass
        return cid, phase, (mech[:3] or [f'targets: {", ".join(targets)}' if targets else ''])
    return None, None, []


def main():
    df = pd.read_csv(TOP20)
    log(f'auditing {len(df)} top killers against perturbation databases\n')
    print(f'{"drug":32} {"rank":>4}  {"L1000 sig":>22}  {"ChEMBL":>12}  known mechanisms')

    rows = []
    terms = load_l1000_terms()
    for _, r in df.iterrows():
        name = str(r['drug_name']).strip()
        nkey = norm(name)
        sig = terms.get(nkey)
        if sig is None:                      # substring fallback, longest first
            cands = [(tn, m) for tn, m in terms.items()
                     if len(tn) >= 6 and (nkey in tn or tn in nkey)]
            if cands:
                sig = max(cands, key=lambda c: len(c[0]))[1]
        cid, phase, mech = chembl_lookup(name)
        row = {'drug_id': r['drug_id'], 'drug_name': name, 'rank': int(r['rank']),
               'l1000_term': sig['term'] if sig else '',
               'l1000_sig_genes': sig['n_genes'] if sig else None,
               'chembl_id': cid or '', 'chembl_max_phase': phase,
               'chembl_mechanisms': '; '.join(mech)}
        rows.append(row)
        l = f"YES ({sig['n_genes']}g)" if sig else '-'
        c = cid or '-'
        print(f'{name:32} {row["rank"]:>4}  {l:>22}  {c:>12}  {(mech[0][:60] if mech else "")}')

    audit = pd.DataFrame(rows)
    audit.to_csv(os.path.join(OUT, 'offtarget_coverage_audit.csv'), index=False)
    json.dump(rows, open(os.path.join(OUT, 'offtarget_coverage_audit.json'), 'w'),
              indent=2)
    n_sig = int(audit['l1000_sig_genes'].notna().sum())
    n_chem = int((audit['chembl_id'] != '').sum())
    log(f'\nsaved offtarget_coverage_audit.csv | '
        f'coverage: {n_sig}/20 with L1000 signatures, {n_chem}/20 with ChEMBL entry')


if __name__ == '__main__':
    main()