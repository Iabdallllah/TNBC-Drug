"""Synergy math and heatmaps for the interactive checkerboard calculator.

Pure numpy/pandas/plotly helpers — imported by dashboard/app.py and tested
directly in tests/test_pipeline.py (no Streamlit runtime needed).

Layout contract
---------------
`viab` is a DataFrame of measured viability (1.0 = fully alive, 0.0 = dead):
  * rows    = concentrations of drug A (first row must be the zero dose),
  * columns = concentrations of drug B (first column must be the zero dose).
The first column therefore carries the drug-A single-agent curve and the first
row the drug-B single-agent curve.

Scores
------
Bliss  : excess = E_obs - (E_A + E_B - E_A*E_B), with observed single-agent
         effects read from the zero-dose row/column.
ZIP    : same combination rule, but E_A/E_B come from 4-parameter logistic fits
         of the single-agent curves (noise-smoothed Bliss; Yadav et al. 2015).
Loewe  : Chou-Talalay combination index, CI = a/A_eq + b/B_eq, where A_eq and
         B_eq are the single-agent doses that produce the observed effect
         (4PL inversion, interpolation fallback). CI < 1 = synergy,
         CI = 1 = additive, CI > 1 = antagonism.
"""
import numpy as np
import pandas as pd
import plotly.express as px

HIGHLIGHT = '#D9480F'
MUTED = '#868E96'
SYN_COLOR = '#2f7d32'
ANT_COLOR = '#b3261e'
ADD_COLOR = '#868E96'


def parse_conc_text(text, max_doses=12):
    """Parse a comma-separated concentration list; returns sorted floats with 0.

    Raises ValueError with a scientist-readable message when the input is not a
    strictly increasing dose series or is too large for the UI.
    """
    raw = [t.strip() for t in str(text).replace(';', ',').split(',') if t.strip()]
    vals = []
    for t in raw:
        try:
            v = float(t)
        except ValueError:
            raise ValueError(f'could not read concentration "{t}" as a number')
        if v < 0:
            raise ValueError('concentrations must be >= 0')
        vals.append(v)
    if not vals:
        raise ValueError('enter at least one concentration per drug')
    vals = sorted(set(vals))
    if vals[0] != 0.0:
        vals = [0.0] + vals
    if len(vals) < 3:
        raise ValueError('need at least two non-zero doses per drug for a checkerboard')
    if len(vals) > max_doses:
        raise ValueError(f'{len(vals)} doses is too many; keep at most {max_doses} per drug')
    return np.asarray(vals, dtype=float)


def _effect(viab):
    return 1.0 - np.asarray(viab, dtype=float)


def single_agent_effects(viab):
    """Drug-A effect curve (first column) and drug-B curve (first row)."""
    e = _effect(viab)
    return e[:, 0], e[0, :]


def placeholder_matrix(conc_a, conc_b, auc_a=None, auc_b=None, hill=1.2):
    """Model-additive null matrix: V_AB = V_A * V_B (Bliss on viability).

    Used ONLY as a labelled placeholder before real plate readings exist. Each
    single-agent curve rises smoothly from 1.0 (zero dose) to 1 - max_effect at
    the top dose, where max_effect = 1 - model AUC when the model scored that
    agent, else 0.55 (clearly a placeholder, not data).
    """
    ca = np.asarray(conc_a, float)
    cb = np.asarray(conc_b, float)

    def curve(conc, auc):
        conc = np.asarray(conc, float)
        if conc.max() <= 0:
            return np.ones_like(conc)
        emax = 0.55 if auc is None or not np.isfinite(auc) else float(np.clip(1 - auc, 0.05, 0.95))
        # A 4PL-shaped curve with ~97% of emax at the top dose. Using the same
        # family the ZIP fit assumes keeps the placeholder additive under all
        # three scores instead of manufacturing a fitting artefact.
        logec50 = np.log10(conc.max()) - 1.5 / hill
        scale = 1.0 + 10.0 ** (-1.5)
        effect = np.zeros_like(conc)
        nz = conc > 0
        effect[nz] = scale * _4pl(np.log10(conc[nz]), 0.0, emax, logec50, hill)
        return 1.0 - np.clip(effect, 0.0, 0.99)

    va = curve(ca, auc_a)
    vb = curve(cb, auc_b)
    return pd.DataFrame(np.outer(va, vb), index=ca, columns=cb)


def _4pl(x, emin, emax, logec50, hill):
    return emin + (emax - emin) / (1.0 + 10.0 ** ((logec50 - x) * hill))


def fit_4pl(conc, effect):
    """Fit a 4PL to a single-agent curve. Returns params or None (fallback)."""
    x = np.asarray(conc, float)
    y = np.asarray(effect, float)
    keep = x > 0
    x, y = np.log10(x[keep]), y[keep]
    if len(x) < 3 or np.nanmax(y) - np.nanmin(y) < 0.05:
        return None
    # sanitise: effects must stay in [0, 1]
    y = np.clip(y, 0.0, 1.0)
    p0 = [float(np.nanmin(y)), float(np.nanmax(y)), float(np.median(x)), 1.0]
    # Weak single agents (max effect < 0.3) are common in this panel; do not
    # reject them by forcing a lower bound above their observed effect.
    lo = [0.0, max(0.0, p0[1] - 0.25), float(x.min()) - 2, 0.3]
    hi = [min(0.2, max(0.01, p0[1] - 0.05)), 1.0, float(x.max()) + 2, 6.0]
    if not (lo[1] < hi[1] and lo[2] < hi[2]):
        return None
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(_4pl, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
        if not np.all(np.isfinite(popt)):
            return None
        return tuple(float(v) for v in popt)
    except Exception:
        return None


def _predict_curve(params, conc):
    conc = np.asarray(conc, float)
    e = np.zeros_like(conc)
    keep = conc > 0
    if params is None:
        return e
    e[keep] = np.clip(_4pl(np.log10(conc[keep]), *params), 0.0, 1.0)
    return e


def _dose_at_effect(params, effect):
    """Invert the 4PL: single-agent dose producing `effect` (0 if unreachable)."""
    if params is None:
        return np.nan
    emin, emax, logec50, hill = params
    effect = float(effect)
    if effect <= emin:
        return 0.0
    if effect >= emax:
        # saturating extrapolation; keep finite so CI stays defined
        return float(10 ** (logec50 + 2.0 / max(hill, 0.3)))
    frac = (effect - emin) / max(emax - emin, 1e-9)
    frac = min(max(frac, 1e-6), 1 - 1e-6)
    log_dose = logec50 - np.log10(1.0 / frac - 1.0) / max(hill, 1e-9)
    return float(10 ** log_dose)


def _summary(excess, conc_a, conc_b):
    e = np.asarray(excess, float)
    combo = e[1:, 1:]
    finite = combo[np.isfinite(combo)]
    return {
        'n_cells': int(finite.size),
        'n_synergistic': int((finite > 0.10).sum()),
        'n_antagonistic': int((finite < -0.10).sum()),
        'frac_synergistic': float((finite > 0.10).mean()) if finite.size else 0.0,
        'mean_excess': float(finite.mean()) if finite.size else 0.0,
        'max_excess': float(finite.max()) if finite.size else 0.0,
        'min_excess': float(finite.min()) if finite.size else 0.0,
    }


def bliss_analysis(viab, conc_a, conc_b):
    e = _effect(viab)
    ea, eb = single_agent_effects(viab)
    expected = ea[:, None] + eb[None, :] - ea[:, None] * eb[None, :]
    excess = e - expected
    return {'observed_effect': e, 'expected': expected, 'excess': excess,
            'summary': _summary(excess, conc_a, conc_b)}


def zip_analysis(viab, conc_a, conc_b):
    e = _effect(viab)
    ea_obs, eb_obs = single_agent_effects(viab)
    pa = fit_4pl(conc_a, ea_obs)
    pb = fit_4pl(conc_b, eb_obs)
    ea = _predict_curve(pa, conc_a)
    eb = _predict_curve(pb, conc_b)
    expected = ea[:, None] + eb[None, :] - ea[:, None] * eb[None, :]
    excess = e - expected
    return {'observed_effect': e, 'expected': expected, 'excess': excess,
            'summary': _summary(excess, conc_a, conc_b),
            'params_a': pa, 'params_b': pb}


def loewe_analysis(viab, conc_a, conc_b):
    e = _effect(viab)
    ea_obs, eb_obs = single_agent_effects(viab)
    pa = fit_4pl(conc_a, ea_obs)
    pb = fit_4pl(conc_b, eb_obs)
    ca = np.asarray(conc_a, float)
    cb = np.asarray(conc_b, float)
    ci = np.full_like(e, np.nan, dtype=float)
    for i, a in enumerate(ca):
        for j, b in enumerate(cb):
            if i == 0 or j == 0:
                ci[i, j] = 1.0
                continue
            eff = e[i, j]
            if not np.isfinite(eff) or eff <= 0:
                continue
            a_eq = _dose_at_effect(pa, eff) if pa is not None else np.nan
            b_eq = _dose_at_effect(pb, eff) if pb is not None else np.nan
            if pa is None or pb is None or not np.isfinite(a_eq) or not np.isfinite(b_eq):
                # interpolation fallback on the observed single-agent curves
                a_eq = _interp_dose(ca[1:], ea_obs[1:], eff)
                b_eq = _interp_dose(cb[1:], eb_obs[1:], eff)
            if not np.isfinite(a_eq) or not np.isfinite(b_eq) or a_eq <= 0 or b_eq <= 0:
                continue
            ci[i, j] = a / a_eq + b / b_eq
    combo = ci[1:, 1:]
    finite = combo[np.isfinite(combo)]
    summary = {
        'n_cells': int(finite.size),
        'median_ci': float(np.median(finite)) if finite.size else np.nan,
        'n_synergy': int((finite < 0.9).sum()),
        'n_additive': int(((finite >= 0.9) & (finite <= 1.1)).sum()),
        'n_antagonism': int((finite > 1.1).sum()),
        'frac_synergy': float((finite < 0.9).mean()) if finite.size else 0.0,
        'n_beyond_range': int((~np.isfinite(combo)).sum()),
    }
    return {'ci': ci, 'summary': summary, 'params_a': pa, 'params_b': pb}


def _interp_dose(conc, effect, target):
    """Monotone interpolation of dose from effect; NaN when out of range."""
    x = np.asarray(conc, float)
    y = np.asarray(effect, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return np.nan
    order = np.argsort(y)
    y, x = y[order], x[order]
    # collapse duplicate effects so np.interp gets a strictly increasing axis
    y_unique, idx = np.unique(y, return_index=True)
    x_unique = x[idx]
    if target < y_unique.min() or target > y_unique.max():
        return np.nan
    return float(np.interp(target, y_unique, x_unique))


def analyse(viab, conc_a, conc_b):
    return {'bliss': bliss_analysis(viab, conc_a, conc_b),
            'zip': zip_analysis(viab, conc_a, conc_b),
            'loewe': loewe_analysis(viab, conc_a, conc_b)}


def _labels(conc, unit):
    out = []
    for v in conc:
        if v == 0:
            out.append('0')
        elif v >= 100:
            out.append(f'{v:g}')
        elif v >= 1:
            out.append(f'{v:g}')
        else:
            out.append(f'{v:g}')
    return [f'{s} {unit}' for s in out]


def _matrix_frame(values, conc_a, conc_b, unit, fmt='.3f'):
    df = pd.DataFrame(np.asarray(values, float),
                      index=_labels(conc_a, unit), columns=_labels(conc_b, unit))
    return df


def heatmap(values, conc_a, conc_b, title, *, colorscale='RdBu_r', zmid=None,
            zmin=None, zmax=None, unit='µM', fmt='.3f', cbar='Score'):
    df = _matrix_frame(values, conc_a, conc_b, unit)
    fig = px.imshow(df, text_auto=fmt, aspect='auto',
                    color_continuous_scale=colorscale,
                    zmin=zmin, zmax=zmax)
    fig.update_traces(textfont=dict(size=9))
    fig.update_layout(title=dict(text=title, font=dict(size=13)),
                      height=430, margin=dict(t=44, b=30, l=10, r=10),
                      xaxis_title='Drug B concentration', yaxis_title='Drug A concentration',
                      coloraxis_colorbar_title=cbar)
    if zmid is not None:
        fig.update_layout(coloraxis_colorbar=dict(tickvals=[zmin, zmid, zmax]))
    fig.update_xaxes(side='bottom')
    return fig


def observed_heatmap(viab, conc_a, conc_b, unit='µM'):
    return heatmap(viab, conc_a, conc_b, 'Observed viability (1 = alive)',
                   colorscale='Viridis', zmin=0, zmax=1, unit=unit, cbar='Viability')


def bliss_heatmap(viab, conc_a, conc_b, unit='µM'):
    a = bliss_analysis(viab, conc_a, conc_b)
    m = float(np.nanmax(np.abs(a['excess']))) or 0.1
    return heatmap(a['excess'], conc_a, conc_b,
                   'Bliss excess (observed − additive expectation)',
                   colorscale='RdBu_r', zmid=0, zmin=-m, zmax=m, unit=unit, cbar='Excess')


def zip_heatmap(viab, conc_a, conc_b, unit='µM'):
    a = zip_analysis(viab, conc_a, conc_b)
    m = float(np.nanmax(np.abs(a['excess']))) or 0.1
    return heatmap(a['excess'], conc_a, conc_b,
                   'ZIP excess (observed − 4PL-expected)',
                   colorscale='RdBu_r', zmid=0, zmin=-m, zmax=m, unit=unit, cbar='Excess')


def loewe_heatmap(viab, conc_a, conc_b, unit='µM'):
    a = loewe_analysis(viab, conc_a, conc_b)
    return heatmap(a['ci'], conc_a, conc_b,
                   'Loewe / Chou-Talalay CI (green < 1 = synergy, red > 1 = antagonism)',
                   colorscale='RdYlGn_r', zmid=1, zmin=0, zmax=2, unit=unit,
                   cbar='CI')


def model_reference_effect(preds, line, drug_a, drug_b):
    """Bliss E_AB from the sealed FerroSelect panel predictions, or None."""
    if preds is None or len(preds) == 0:
        return None
    p = preds[preds['line'] == line].set_index('drug')['pred_auc'].astype(float)
    if drug_a not in p.index or drug_b not in p.index:
        return None
    ea, eb = 1.0 - p[drug_a], 1.0 - p[drug_b]
    return {'auc_a': float(p[drug_a]), 'auc_b': float(p[drug_b]),
            'e_a': float(ea), 'e_b': float(eb),
            'e_ab': float(ea + eb - ea * eb)}


def comparison_chart(ref_effect, observed_effect):
    """Two bars: model Bliss expectation vs observed effect at the top doses."""
    rows = [{'source': 'Model (Bliss, sealed predictions)', 'effect': ref_effect},
            {'source': 'Observed (highest-dose cell)', 'effect': observed_effect}]
    d = pd.DataFrame(rows)
    fig = px.bar(d, x='effect', y='source', orientation='h',
                 color='source',
                 color_discrete_sequence=['#495057', HIGHLIGHT],
                 labels={'effect': 'Killing effect (1 − viability)'})
    fig.update_traces(texttemplate='%{x:.3f}', textposition='outside')
    fig.update_layout(height=220, showlegend=False, margin=dict(t=10, b=30, l=10),
                      yaxis_title=None,
                      xaxis=dict(range=[0, max(1.0, max(rows[0]['effect'] or 0,
                                                        rows[1]['effect'] or 0) * 1.2)]))
    return fig
