"""TNBC drug-sensitivity prediction API (PRD FR-08).

Serves the trained CatBoost model:
  GET  /health
  GET  /model-info
  POST /predict     {cell_line_id, drug_id}            -> predicted_AUC
  POST /rank        {cell_line_id, top_k}               -> ranked drug list
  POST /rank-pairs  {top_singles, top_pairs}           -> Bliss combo ranking (MDA-MB-231 vs MCF7)

Run:  uvicorn src.api.main:app --host 127.0.0.1 --port 8000
Data dir resolution: $TNBC_DATA_DIR or ./outputs_local or ./Data
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_VERSION = 'catboost-v1.0'
DISCLAIMER = 'predicted sensitivity in cell model — NOT a clinical recommendation (PRD §5.2)'
MDA_ID = 'ACH-000768'   # MDA-MB-231 (TNBC)
MCF_ID = 'ACH-000019'   # MCF7 (Luminal control)


def _data_dirs():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cands = [os.environ.get('TNBC_DATA_DIR', ''),
             os.path.join(here, 'outputs_local'),
             os.path.join(here, 'Data'), '/content']
    return [d for d in cands if d and os.path.isdir(d)]


def _find(name):
    for d in _data_dirs():
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'{name} not found in { _data_dirs()}')


@lru_cache(maxsize=1)
def _bundle():
    import pyarrow.parquet as pq
    cols = pq.read_schema(_find('tnbc_ml_dataset.parquet')).names
    feat = [c for c in cols
            if c.startswith('pca_') or (c.startswith('drug_')
                                        and c not in ('drug_id', 'drug_name'))]
    ml_ids = pd.read_parquet(_find('tnbc_ml_dataset.parquet'), columns=['drug_id'])
    pos = {c: i for i, c in enumerate(feat)}
    pca_cols = [c for c in feat if c.startswith('pca_')]
    drug_ids = sorted(ml_ids['drug_id'].astype(str).unique())
    assert all(f'drug_{d}' in pos for d in drug_ids), 'one-hot/model feature mismatch'
    pca = pd.read_parquet(_find('expression_pca128_breast.parquet'))
    pca = pca.drop_duplicates(subset=['ModelID'], keep='first').reset_index(drop=True)
    model = CatBoostRegressor()
    model.load_model(_find('catboost_tnbc_v1.cbm'))
    names = {}
    try:
        cmap = pd.read_csv(_find('REPURPOSINGLog2ViabilityCollapsedConditions.csv'),
                           usecols=['CompoundName', 'CompoundID']).drop_duplicates('CompoundID')
        names = dict(zip(cmap['CompoundID'].astype(str), cmap['CompoundName']))
    except (FileNotFoundError, ValueError):
        pass
    return {'feat': feat, 'pos': pos, 'pca_cols': pca_cols, 'drug_ids': drug_ids,
            'pca': pca, 'model': model, 'names': names}


def _vec(model_id: str, drug_id: str):
    b = _bundle()
    row = b['pca'].loc[b['pca']['ModelID'] == model_id]
    if len(row) != 1:
        raise HTTPException(404, f'Unknown/no PCA for {model_id}')
    col = f'drug_{drug_id}'
    if col not in b['pos']:
        raise HTTPException(404, f'Unknown drug {drug_id}')
    x = np.zeros((1, len(b['feat'])), dtype=np.float32)
    for c in b['pca_cols']:
        x[0, b['pos'][c]] = float(row[c].iloc[0])
    x[0, b['pos'][col]] = 1.0
    return x


def _auc(model_id: str, drug_id: str) -> float:
    b = _bundle()
    return float(np.clip(b['model'].predict(_vec(model_id, drug_id))[0], 0, 1))


app = FastAPI(title='TNBC Drug Response Predictor', version=MODEL_VERSION)


class PredictIn(BaseModel):
    cell_line_id: str
    drug_id: str


class RankIn(BaseModel):
    cell_line_id: str = MDA_ID
    top_k: int = 20


class PairsIn(BaseModel):
    top_singles: int = 20
    top_pairs: int = 5


@app.get('/health')
def health():
    return {'ok': True, 'model_version': MODEL_VERSION}


@app.get('/model-info')
def model_info():
    b = _bundle()
    return {'model_version': MODEL_VERSION,
            'features': len(b['feat']),
            'cell_lines_with_pca': int(b['pca']['ModelID'].nunique()),
            'drugs': len(b['drug_ids']),
            'disclaimer': DISCLAIMER}


@app.post('/predict')
def predict(inp: PredictIn):
    return {'cell_line_id': inp.cell_line_id, 'drug_id': inp.drug_id,
            'predicted_auc': _auc(inp.cell_line_id, inp.drug_id),
            'model_version': MODEL_VERSION, 'disclaimer': DISCLAIMER}


@app.post('/rank')
def rank(inp: RankIn):
    b = _bundle()
    rows = []
    for d in b['drug_ids']:
        rows.append({'drug_id': d, 'drug_name': b['names'].get(d, d),
                     'predicted_auc': _auc(inp.cell_line_id, d)})
    rows.sort(key=lambda r: r['predicted_auc'])
    out = []
    for i, r in enumerate(rows[:max(1, inp.top_k)], 1):
        r = dict(r)
        r['rank'] = i
        r['call'] = 'SENSITIVE (AUC<0.3)' if r['predicted_auc'] < 0.3 else 'resistant/intermediate'
        out.append(r)
    return {'cell_line_id': inp.cell_line_id, 'model_version': MODEL_VERSION,
            'ranking': out, 'disclaimer': DISCLAIMER}


@app.post('/rank-pairs')
def rank_pairs(inp: PairsIn):
    b = _bundle()
    rm = b['pca'].loc[b['pca']['ModelID'] == MDA_ID]
    rc = b['pca'].loc[b['pca']['ModelID'] == MCF_ID]
    if len(rm) != 1 or len(rc) != 1:
        raise HTTPException(404, 'MDA-MB-231/MCF7 PCA rows missing')
    xm = np.tile(rm[b['pca_cols']].to_numpy(dtype=np.float32), (len(b['drug_ids']), 1))
    xc = np.tile(rc[b['pca_cols']].to_numpy(dtype=np.float32), (len(b['drug_ids']), 1))
    Xm = np.zeros((len(b['drug_ids']), len(b['feat'])), dtype=np.float32)
    Xc = np.zeros((len(b['drug_ids']), len(b['feat'])), dtype=np.float32)
    for j, d in enumerate(b['drug_ids']):
        Xm[j, [b['pos'][c] for c in b['pca_cols']]] = xm[j]
        Xc[j, [b['pos'][c] for c in b['pca_cols']]] = xc[j]
        Xm[j, b['pos'][f'drug_{d}']] = 1.0
        Xc[j, b['pos'][f'drug_{d}']] = 1.0
    am = np.clip(b['model'].predict(Xm), 0, 1)
    ac = np.clip(b['model'].predict(Xc), 0, 1)
    order = np.argsort(am)[:max(2, inp.top_singles)]
    ids = [b['drug_ids'][i] for i in order]
    em, ec = 1 - am[order], 1 - ac[order]
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            A, B = ids[i], ids[j]
            e_mda = float(em[i] + em[j] - em[i] * em[j])
            e_mcf = float(ec[i] + ec[j] - ec[i] * ec[j])
            pairs.append({'drugA_id': A, 'drugA_name': b['names'].get(A, A),
                          'drugB_id': B, 'drugB_name': b['names'].get(B, B),
                          'E_AB_MDA231': round(e_mda, 4), 'E_AB_MCF7': round(e_mcf, 4),
                          'selectivity_gap': round(e_mda - e_mcf, 4)})
    pairs.sort(key=lambda r: r['selectivity_gap'], reverse=True)
    out = []
    for i, r in enumerate(pairs[:max(1, inp.top_pairs)], 1):
        r = dict(r)
        r['pair_rank'] = i
        out.append(r)
    return {'model_version': MODEL_VERSION, 'method': 'Bliss independence (additivity assumption — validate in vitro)',
            'pairs': out, 'disclaimer': DISCLAIMER}
