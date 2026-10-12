"""Bliss-based 2-drug combo ranking (MY_OUTLINE §6B, stretch).

Method (NO new training — additivity assumption, must be validated in vitro):
  E      = 1 - predicted_AUC            (per line)
  E_AB   = E_A + E_B - E_A*B            (Bliss expected, per line)
  gap    = E_AB(MDA-MB-231) - E_AB(MCF7)  (selectivity-aware rank key)

Pairs come from the Top-N singles only (N=20 -> max 190 pairs).
Optional --targets CSV (drug_id,target,pathway) enables the distinct-pathway
filter; without it all pairs are ranked and flagged for Bioinfo review.

Outputs (outputs_local/):
  combo_pairs_full.csv  all pairs
  combo_rankings.csv    Top-K pairs for wet-lab handoff (max 2 pairs per outline)
"""
import argparse
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, 'outputs_local')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-singles', type=int, default=20)
    ap.add_argument('--top-pairs', type=int, default=5)
    ap.add_argument('--targets', default=None,
                    help='optional CSV with drug_id,target,pathway columns')
    ap.add_argument('--outdir', default=OUT)
    a = ap.parse_args()

    rank = pd.read_csv(os.path.join(a.outdir, 'mda_mb_231_full_ranking.csv'))
    top = rank.head(a.top_singles).reset_index(drop=True)
    print(f'singles pool: {len(top)} drugs -> max {len(top) * (len(top) - 1) // 2} pairs')

    tmap = {}
    if a.targets and os.path.exists(a.targets):
        t = pd.read_csv(a.targets)
        tmap = dict(zip(t['drug_id'].astype(str),
                        zip(t.get('target', ''), t.get('pathway', ''))))
        print(f'target map: {len(tmap)} drugs (distinct-pathway filter ON)')
    else:
        print('no target map -> pathway filter PENDING Bioinfo review')

    rows = []
    ids = top['drug_id'].astype(str).tolist()
    rec = top.set_index(top['drug_id'].astype(str))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            A, B = ids[i], ids[j]
            ra, rb = rec.loc[A], rec.loc[B]
            ea_mda, eb_mda = 1 - ra['pred_AUC_MDA231'], 1 - rb['pred_AUC_MDA231']
            ea_mcf, eb_mcf = 1 - ra['pred_AUC_MCF7'], 1 - rb['pred_AUC_MCF7']
            e_mda = ea_mda + eb_mda - ea_mda * eb_mda
            e_mcf = ea_mcf + eb_mcf - ea_mcf * eb_mcf
            pa, pb = tmap.get(A, ('?', '?'))[1], tmap.get(B, ('?', '?'))[1]
            rows.append({
                'drugA_id': A, 'drugA_name': ra['drug_name'],
                'drugB_id': B, 'drugB_name': rb['drug_name'],
                'E_AB_MDA231': round(float(e_mda), 4),
                'E_AB_MCF7': round(float(e_mcf), 4),
                'selectivity_gap': round(float(e_mda - e_mcf), 4),
                'pathwayA': tmap.get(A, ('?', '?'))[1],
                'pathwayB': tmap.get(B, ('?', '?'))[1],
                'pathway_check': 'distinct' if (pa != '?' and pa != pb)
                                 else ('PENDING-Bioinfo' if not tmap else 'same-pathway'),
            })
    pairs = pd.DataFrame(rows).sort_values('selectivity_gap', ascending=False).reset_index(drop=True)
    pairs.index = pairs.index + 1
    pairs.to_csv(os.path.join(a.outdir, 'combo_pairs_full.csv'), index_label='pair_rank')
    print(f'saved combo_pairs_full.csv ({len(pairs)} pairs)')

    topk = pairs.head(a.top_pairs).copy()
    topk.to_csv(os.path.join(a.outdir, 'combo_rankings.csv'), index_label='pair_rank')
    print(f'saved combo_rankings.csv (Top-{a.top_pairs})')
    print()
    print(topk[['drugA_name', 'drugB_name', 'E_AB_MDA231', 'E_AB_MCF7',
                'selectivity_gap', 'pathway_check']].to_string())
    print()
    print('NOTE: Bliss additivity assumption — synergy/additive/antagonistic verdict '
          'requires 4x4 checkerboard validation (outline §9). Max 2 pairs to wet-lab.')


if __name__ == '__main__':
    main()
