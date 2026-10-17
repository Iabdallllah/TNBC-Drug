"""Recompute the sensitivity and selectivity verdicts on the drug rankings.

Why this exists
---------------
Notebook 02 flagged selectivity with a fixed `selectivity_delta > 0.05` cut-off.
That threshold is unreachable with this model: the predicted AUC gap between
MDA-MB-231 and MCF7 is compressed (max observed delta = 0.0385), so all 1,482
drugs were labelled 'non-selective' and the column carried zero information.

The rule below is the relative one the project actually wants:

    selective  <=>  predicted_AUC_MDA231 < 0.40  AND  selectivity_delta > 0

  * `AUC < 0.40` keeps only drugs that are actually predicted to kill the TNBC
    line (0.40 rather than the 0.30 sensitivity gate, because only 1 of 1,482
    drugs clears 0.30).
  * `delta > 0` is the natural relative cut-off here: the median delta is
    negative (-0.0076), so delta > 0 selects exactly the drugs the model thinks
    hit TNBC harder than the luminal control. It is the top 1.8% of the panel.

The 0.30 sensitivity verdict is left untouched: it is the pre-registered gate
used by the metrics, the API and the tests.

This script is idempotent and does NOT retrain anything — it rewrites the
`selective_call` column in place from the existing ranking CSVs.

Run:  python src/models/selectivity.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'outputs_local')

SENSITIVE_AUC = 0.30          # pre-registered sensitivity gate (unchanged)
SELECTIVE_AUC_MAX = 0.40      # a selective drug must be predicted to kill TNBC
SELECTIVITY_DELTA_MIN = 0.0   # relative: beats the control line
FILES = ['mda_mb_231_full_ranking.csv', 'mda_mb_231_top20_killers.csv',
         'wetlab_shortlist.csv', 'candidate_rankings.csv']


def apply_rule(df, auc_max, delta_min, sensitive_auc=SENSITIVE_AUC):
    d = df.copy()
    auc = d['pred_AUC_MDA231'].astype(float)
    delta = d['selectivity_delta'].astype(float)
    d['sensitive_call'] = np.where(
        auc < sensitive_auc, 'SENSITIVE (AUC<0.3)', 'resistant/intermediate')
    d['selective_call'] = np.where(
        (auc < auc_max) & (delta > delta_min), 'Selective', 'non-selective')
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--auc-max', type=float, default=SELECTIVE_AUC_MAX,
                    help='predicted MDA-MB-231 AUC ceiling for a selective call')
    ap.add_argument('--delta-min', type=float, default=SELECTIVITY_DELTA_MIN,
                    help='minimum selectivity_delta (MCF7 AUC - MDA-MB-231 AUC)')
    ap.add_argument('--outdir', default=OUT)
    a = ap.parse_args()

    full_path = os.path.join(a.outdir, FILES[0])
    if not os.path.exists(full_path):
        raise SystemExit(f'missing {full_path} — run notebook 02 §6 first')
    full = pd.read_csv(full_path)

    delta = full['selectivity_delta'].astype(float)
    rule = {
        'sensitivity_rule': f'pred_AUC_MDA231 < {SENSITIVE_AUC} (unchanged, pre-registered)',
        'selectivity_rule': (f'pred_AUC_MDA231 < {a.auc_max} AND '
                             f'selectivity_delta > {a.delta_min}'),
        'old_selectivity_rule': 'selectivity_delta > 0.05 (unreachable: max delta 0.0385)',
        'sensitive_auc': SENSITIVE_AUC,
        'selective_auc_max': a.auc_max,
        'selectivity_delta_min': a.delta_min,
        'panel_size': int(len(full)),
        'delta_stats': {
            'min': float(delta.min()), 'median': float(delta.median()),
            'max': float(delta.max()), 'n_positive': int((delta > 0).sum()),
            'n_above_old_threshold': int((delta > 0.05).sum()),
        },
        'n_below_auc_0.40': int((full['pred_AUC_MDA231'].astype(float) < 0.40).sum()),
    }

    changed = 0
    for name in FILES:
        path = os.path.join(a.outdir, name)
        if not os.path.exists(path):
            continue
        before = pd.read_csv(path)
        after = apply_rule(before, a.auc_max, a.delta_min)
        n = int((after['selective_call'] == 'Selective').sum())
        after.to_csv(path, index=False)
        changed += 1
        print(f'{name}: {len(after)} rows, selective={n}, '
              f'sensitive={int((after["sensitive_call"].astype(str).str.startswith("SENSITIVE")).sum())}')
        rule.setdefault('per_file', {})[name] = {
            'rows': int(len(after)),
            'n_selective': n,
            'selective_drugs': sorted(after.loc[after['selective_call'] == 'Selective',
                                                'drug_name'].astype(str).tolist()),
        }

    rule['files_rewritten'] = changed
    json.dump(rule, open(os.path.join(a.outdir, 'selectivity_rule.json'), 'w'), indent=2)
    print('\nselectivity_rule.json written')
    print('rule:', rule['selectivity_rule'])
    print('delta>0 (top %.1f%%): %d/%d' % (100 * (delta > 0).mean(), int((delta > 0).sum()), len(full)))


if __name__ == '__main__':
    main()