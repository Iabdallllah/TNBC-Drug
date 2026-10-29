"""Interactive protein rescue-network for the dashboard.

Builds a Plotly network from the expressed-protein signatures:
  * killer  -> suppressed proteins (its Down signature ∩ expressed genes),
  * rescue  -> the validated killer-Down ∩ rescue-Up overlap (the reversal
               evidence), because the full rescue-Up set is not stored in
               `protein_array_shortlist.csv` (that file is a Down-signature
               shortlist for the array, not a rescue-Up table),
  * shared  -> proteins in both sets (every drawn rescue node is shared).

Data sources (outputs_local/):
  * protein_array_shortlist.csv       full killer Down gene lists
  * offtarget_rescue_reversal.csv     validated overlap + hypergeometric p
  * offtarget_coverage_ferroselect.csv L1000 term provenance per panel drug

The provenance check is deliberately strict. A legacy fuzzy matcher resolved
the abbreviation "NAC" inside "SolifeNACin-Succinate"; that term is a different
molecule, so NAC is withheld instead of being plotted under a false label.
"""
import re

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

KILLER_COLOR = '#b3261e'
RESCUE_COLOR = '#1c7ed6'
SHARED_COLOR = '#2f7d32'
SUPP_COLOR = '#e8590c'
DRUG_KILLER_COLOR = '#d9480f'
DRUG_RESCUE_COLOR = '#1971c2'
MUTED = '#868E96'

KILLER_OPTIONS = ['RSL3 + SIMVASTATIN', 'ERASTIN + SIMVASTATIN',
                  'SULFASALAZINE + SIMVASTATIN', 'RSL3', 'ERASTIN',
                  'SULFASALAZINE', 'SIMVASTATIN']
RESCUE_OPTIONS = ['DEFEROXAMINE', 'N-ACETYLCYSTEINE']


def _norm(name):
    """Compare drug labels ignoring case and spaces ('RSL3+SIMVASTATIN' vs 'RSL3 + SIMVASTATIN')."""
    return str(name).replace(' ', '').upper()


def _base_term(term):
    """Strip the trailing ' Up' / ' Down' from an L1000 term and normalise it."""
    return re.sub(r'[^a-z0-9]+', '',
                  re.sub(r'\s+(up|down)$', '', str(term), flags=re.I).lower())


def signature_audit(rescue, coverage=None):
    """Return (valid, l1000_term, message) for one rescue agent.

    A rescue agent may only be plotted when its L1000 term actually names that
    molecule. Short synonyms such as "NAC" must never be substring-matched
    against unrelated words ("SolifeNACin"); that is how the legacy pipeline
    produced a Solifenacin signature labelled as NAC.
    """
    if coverage is None or not len(coverage) or 'panel_drug' not in coverage.columns:
        return True, None, ''
    sub = coverage[coverage['panel_drug'].map(_norm) == _norm(rescue)]
    if not len(sub):
        return True, None, ''
    term = str(sub.iloc[0].get('l1000_term') or '').strip()
    if not term or term.lower() == 'nan':
        return False, term, (f'{rescue}: no L1000 consensus signature is available. '
                             'The rescue network is withheld rather than guessed.')
    keys = {_norm(rescue)}
    if rescue.upper().startswith('N-ACETYL'):
        keys |= {'nacetylcysteine', 'nacetyllcysteine', 'acetylcysteine'}
    if rescue.upper().startswith('DEFEROX') or rescue.upper().startswith('DESFER'):
        keys |= {'deferoxamine', 'desferrioxamine', 'deferioxamine'}
    base = _base_term(term)
    if not any(len(k) >= 6 and (k in base or base in k) for k in keys):
        return False, term, (
            f'{rescue} is backed by the L1000 term “{term}”, which is a different '
            'molecule (legacy fuzzy-match artefact: the abbreviation “NAC” matched '
            'inside “SolifeNACin”). The rescue network is withheld until a genuine '
            f'{rescue} signature is supplied.')
    return True, term, ''


def _genes(df, drug):
    if df is None or 'gene' not in df.columns or 'drug' not in df.columns:
        return set()
    sub = df[df['drug'].map(_norm) == _norm(drug)]
    return set(sub['gene'].dropna().astype(str))


def _event_count(shortlist, panel_map, drug, line):
    """Suppression events for one member: shortlist first, panel map for single statins."""
    n = len(_genes(shortlist, drug))
    if n:
        return n
    if panel_map is not None and 'hit_genes' in panel_map.columns:
        sub = panel_map[(panel_map['drug'].map(_norm) == _norm(drug))
                        & (panel_map['line'] == line)
                        & (panel_map['direction'].astype(str).str.lower() == 'down')]
        if len(sub):
            v = pd.to_numeric(sub['hit_genes'], errors='coerce').max()
            if pd.notna(v):
                return int(v)
    return 0


def _reversal_rows(reversal, killer, rescue, line):
    """Rows for each single-agent member of the killer arm, for one rescue agent."""
    if reversal is None or not len(reversal) or 'line' not in reversal.columns:
        return None
    parts = [p.strip() for p in killer.split('+')]
    sub = reversal[(reversal['line'] == line)
                   & (reversal['killer'].map(_norm).isin([_norm(p) for p in parts]))
                   & (reversal['rescue'].map(_norm) == _norm(rescue))]
    return sub if len(sub) else None


def _overlap_genes(rows):
    """Union of the stored killer-Down ∩ rescue-Up genes (validated overlap)."""
    genes = set()
    if rows is not None and 'genes' in rows.columns:
        for value in rows['genes'].dropna().astype(str):
            genes.update(g.strip() for g in value.split(',') if g.strip())
    return genes


def build_sets(shortlist, reversal, killer, rescue, line, panel_map=None,
               coverage=None):
    """Return the suppressed / validated-overlap sets plus the overlap stats."""
    suppressed = _genes(shortlist, killer)
    valid, term, warning = signature_audit(rescue, coverage)
    rows = _reversal_rows(reversal, killer, rescue, line)
    rescued = _overlap_genes(rows) if valid else set()
    shared = suppressed & rescued if valid else set()

    p = enrich = None
    n_rescue_signature = 0
    if valid and rows is not None:
        p = float(rows['reversal_p'].min())
        enrich = float(rows['reversal_enrich'].max())
        if 'rescue_up_genes' in rows.columns:
            n_rescue_signature = int(pd.to_numeric(
                rows['rescue_up_genes'], errors='coerce').max())

    note = ''
    if ' + ' in killer:
        parts = [x.strip() for x in killer.split('+')]
        counts = {part: _event_count(shortlist, panel_map, part, line) for part in parts}
        total_events = sum(counts.values())
        if all(counts.values()) and total_events >= len(suppressed):
            note = (f"{len(suppressed)} unique suppressed proteins "
                    f"({total_events} suppression events: "
                    + ' + '.join(f'{v} {k}' for k, v in counts.items())
                    + f"; {total_events - len(suppressed)} shared between members)")
        else:
            note = f'{len(suppressed)} unique suppressed proteins for {killer}'
    elif suppressed:
        note = f'{len(suppressed)} suppressed proteins for {killer} on the shortlist'
    if valid and rows is not None:
        note += (f'; rescue-up signature: {n_rescue_signature} genes, '
                 f'{len(rescued)} validated overlap genes stored')
    return {
        'suppressed': suppressed, 'rescued': rescued, 'shared': shared,
        'n_suppressed': len(suppressed), 'n_rescued': n_rescue_signature,
        'n_overlap': len(rescued), 'n_shared': len(shared),
        'p': p, 'enrich': enrich, 'note': note, 'base_killer': killer.split('+')[0].strip(),
        'valid': valid, 'signature_term': term, 'warning': warning,
    }


def _pick_nodes(info, max_nodes):
    """Shared genes first, then a proportional sample of suppressed and rescued nodes.

    A pure 'suppressed first' cut hid every re-expressed protein whenever the killer
    list was larger than the node budget, which defeats the point of the role-coloured
    graph. Keep the shared rescue evidence, then show both roles.
    """
    limit = max(4, int(max_nodes))
    shared = sorted(info['shared'])[:limit]
    supp = sorted(info['suppressed'] - info['shared'])
    resc = sorted(info['rescued'] - info['shared'])
    remaining = limit - len(shared)
    if remaining <= 0 or not (supp or resc):
        return shared
    total = len(supp) + len(resc)
    n_s = int(round(remaining * len(supp) / total)) if total else 0
    n_s = min(len(supp), max(1 if supp else 0, n_s))
    n_r = min(len(resc), remaining - n_s)
    n_s = min(len(supp), remaining - n_r)
    return shared + supp[:n_s] + resc[:n_r]


def rescue_network_figure(shortlist, reversal, killer, rescue, line,
                          max_nodes=80, label_shared=True, seed=42, panel_map=None,
                          coverage=None):
    """Plotly figure + summary dict for one killer x rescue pair on one line."""
    info = build_sets(shortlist, reversal, killer, rescue, line, panel_map=panel_map,
                      coverage=coverage)
    if not info['valid']:
        return None, info
    genes = _pick_nodes(info, max_nodes)
    gset = set(genes)

    G = nx.Graph()
    G.add_node(killer, kind='killer')
    G.add_node(rescue, kind='rescue')
    for g in genes:
        if g in info['shared']:
            kind = 'shared'
        elif g in info['suppressed']:
            kind = 'suppressed'
        else:
            kind = 'rescued'
        G.add_node(g, kind=kind)

    for g in genes:
        if g in info['suppressed']:
            G.add_edge(killer, g, etype='suppresses')
        if g in info['rescued']:
            G.add_edge(rescue, g, etype='re-expresses')

    pos = nx.spring_layout(G, seed=seed, k=2.6 / np.sqrt(max(len(G), 2)) + 0.22,
                           iterations=120)
    xs = np.array([pos[n][0] for n in G.nodes()])
    ys = np.array([pos[n][1] for n in G.nodes()])
    if xs.max() > xs.min():
        xs = (xs - xs.min()) / (xs.max() - xs.min()) * 2 - 1
    if ys.max() > ys.min():
        ys = (ys - ys.min()) / (ys.max() - ys.min()) * 2 - 1
    xy = {n: (float(xs[i]), float(ys[i])) for i, n in enumerate(G.nodes())}

    def edge_trace(etype, color, dash):
        ex, ey = [], []
        for a, b, d in G.edges(data=True):
            if d['etype'] != etype:
                continue
            (x0, y0), (x1, y1) = xy[a], xy[b]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
        return go.Scatter(x=ex, y=ey, mode='lines', line=dict(color=color, width=1,
                                                              dash=dash),
                          hoverinfo='none', showlegend=False)

    fig = go.Figure()
    fig.add_trace(edge_trace('suppresses', KILLER_COLOR, 'dot'))
    fig.add_trace(edge_trace('re-expresses', RESCUE_COLOR, 'solid'))

    roles = {
        'killer': (DRUG_KILLER_COLOR, 22, 'Killer drug'),
        'rescue': (DRUG_RESCUE_COLOR, 22, 'Rescue control'),
        'shared': (SHARED_COLOR, 10, 'Validated overlap (killer↓ ∩ rescue↑)'),
        'suppressed': (SUPP_COLOR, 7, 'Suppressed by killer'),
        'rescued': (RESCUE_COLOR, 7, 'Rescue-up signature (not stored)'),
    }
    for kind, (color, size, legend) in roles.items():
        nodes = [n for n in G.nodes() if G.nodes[n]['kind'] == kind]
        if not nodes:
            continue
        text = []
        for n in nodes:
            if kind in ('killer', 'rescue'):
                text.append(n)
            elif kind == 'shared':
                text.append(n if label_shared else '')
            else:
                text.append('')
        hover = []
        for n in nodes:
            if kind in ('killer', 'rescue'):
                hover.append(f'<b>{n}</b><br>Role: {legend}')
            else:
                extra = info.get('array_friendly', {}).get(n, '')
                hover.append(f'<b>{n}</b><br>Role: {legend}' + (f'<br>{extra}' if extra else ''))
        fig.add_trace(go.Scatter(
            x=[xy[n][0] for n in nodes], y=[xy[n][1] for n in nodes],
            mode='markers+text', text=text, textposition='top center',
            textfont=dict(size=9, color='#343A40'),
            marker=dict(size=size, color=color, line=dict(width=1, color='#ffffff')),
            customdata=hover, hovertext=hover, hoverinfo='text',
            name=legend, showlegend=True))

    fig.update_layout(
        height=620, margin=dict(t=30, b=20, l=10, r=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    xanchor='left', x=0, font=dict(size=10)),
        hoverlabel=dict(bgcolor='#ffffff'))
    info['shown'] = len(genes)
    info['_graph'] = G
    return fig, info


def mechanism_path_figure(paths_json, drugs):
    """Curated drug -> target -> pathway -> GPX4 -> ferroptosis routes."""
    if not paths_json:
        return None, {}
    routes = paths_json.get('shortest_to_ferroptosis', {})
    nodes, edges = {}, []
    for drug in drugs:
        path = routes.get(drug)
        if not path:
            continue
        for a, b in zip(path, path[1:]):
            if (a, b) not in edges:
                edges.append((a, b))
            nodes.setdefault(a, 'drug' if a == drug else 'node')
            nodes.setdefault(b, 'drug' if b == drug else 'node')
    for n in nodes:
        if n == 'GPX4':
            nodes[n] = 'gpx4'
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = nx.spring_layout(G.to_undirected(), seed=7, k=0.9)
    fig = go.Figure()
    for a, b in edges:
        fig.add_trace(go.Scatter(x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]],
                                 mode='lines+markers', marker=dict(size=5, color=MUTED),
                                 line=dict(color=MUTED, width=1.2),
                                 hoverinfo='none', showlegend=False))
    palette = {'drug': '#d9480f', 'gpx4': '#2f7d32', 'node': '#495057'}
    for kind, color in palette.items():
        ns = [n for n in nodes if nodes[n] == kind]
        if not ns:
            continue
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns],
                                 mode='markers+text', text=ns, textposition='top center',
                                 textfont=dict(size=9), marker=dict(size=14, color=color),
                                 name=kind, hovertext=ns, hoverinfo='text'))
    fig.update_layout(height=520, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(t=20, b=10, l=10, r=10), showlegend=False)
    return fig, nodes
