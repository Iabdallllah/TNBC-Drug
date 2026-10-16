"""Generate the sealable prediction document (proposal §6.2, Objective 4).

Reads F2-phase-1 outputs + support audit, writes outputs_local/sealed_prediction.md:
  - frozen model identity, dataset versions, timestamp (UTC)
  - predicted ranking per agent across the 3 proposal lines
  - nominated statin pairing (simvastatin: most training rows) with rationale
  - SHA256 of the predictions CSV for tamper evidence
  - sign-off block (computational pair + supervisor + date)

The SIGNING itself (print, sign, lodge before first plate) is a human step.
"""
import datetime
import hashlib
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, 'outputs_local')


def main():
    res = json.load(open(os.path.join(OUT, 'ferroselect_results.json')))
    pred = pd.read_csv(os.path.join(OUT, 'ferroselect_panel_predictions.csv'))
    now = datetime.datetime.now(datetime.timezone.utc)
    sha = hashlib.sha256(open(os.path.join(
        OUT, 'ferroselect_panel_predictions.csv'), 'rb').read()).hexdigest()[:16]

    lines = [
        '# Sealed prediction — FerroSelect computational half',
        '',
        f'Date (UTC): {now.strftime("%Y-%m-%d %H:%M")}  |  Model: {res["frozen_model"]} '
        f'(FerroSelect track, nested cell-line CV)  |  Seed: {res["seed"]}',
        f'Dataset: DepMap 26Q3 + PRISM-REPURPOSING (n={res["n_panel_rows"]} panel rows, '
        f'{res["n_lines"]} lines)  |  Predictions SHA256: `{sha}`',
        '',
        '## Predicted sensitivity (AUC; lower = more sensitive)',
        '',
        '| Line | Agent | Predicted AUC | Training rows | Confidence |',
        '|---|---|---|---|---|',
    ]
    for _, r in pred.iterrows():
        lines.append(f"| {r['line']} | {r['drug']} | {r['pred_auc']:.4f} "
                     f"| {r['n_train_rows']} | {r['confidence']} |")
    sp = res['per_model_oof'][res['frozen_model']]['spearman']
    lines += [
        '',
        f"Frozen-model pooled OOF Spearman: {sp:.4f} "
        f"(ceiling check: {res['ceiling_flag']})",
        '',
        '## Nominated statin pairing',
        '',
        '**Simvastatin (fixed sub-lethal dose, ~IC20 set in pilot) + sulfasalazine dose-response**, '
        'on the least sensitive line per the table above. Rationale: simvastatin has the most '
        'training rows (22) of the statins; sulfasalazine is the cheap Type-I workhorse. '
        'This nominates the pairing to TEST — it does not claim predicted synergy.',
        '',
        '## Sign-off (human step — print, sign, lodge BEFORE first plate is seeded)',
        '',
        '- Computational pair: ______________________  Date: __________',
        '- Supervisor (holds sealed copy + relabels agents A–H): ______________________  Date: __________',
        '',
        '_Any hyperparameter, feature-set, or training-data change after this timestamp '
        'voids the seal._',
    ]
    p = os.path.join(OUT, 'sealed_prediction.md')
    open(p, 'w').write('\n'.join(lines) + '\n')
    print('wrote', p)


if __name__ == '__main__':
    main()
