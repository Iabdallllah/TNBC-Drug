"""FerroSelect network pharmacology (F3).

Small, curated drug-target-pathway graph encoding proposal §2.2 + Figure 2:
  sulfasalazine/erastin -> SLC7A11 (System xc-) -| cystine -> GSH -> GPX4
  simvastatin/lovastatin -> HMGCR -| mevalonate -> IPP -> seleno-tRNA -> GPX4 synthesis
  RSL3 -> GPX4 (direct; flagged: no training rows in our matrices)
  rescue controls: ferrostatin-1, deferoxamine, NAC (rescue) vs Z-VAD-FMK (must NOT rescue)

Core ferroptosis genes attached per proposal text (GPX4, SLC7A11, GCLC, GSS,
ACSL4, LPCAT3, NFE2L2, FTH1, TFRC, SLC3A2).

Source flag: CURATED_FROM_PROPOSAL (refs [1-3,9,18] therein). Full
FerrDb/STRING enrichment is a documented follow-up, not claimed here.

Analyses (networkx, no external calls):
  1. shortest directed paths: each drug -> GPX4 and each drug -> ferroptosis
  2. hub ranking by total degree
  3. proximity score between the sulfasalazine and statin target neighborhoods
Outputs: ferroselect_network.png, network_table.csv, network_paths.json
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, 'outputs_local')

FERROPTOSIS = 'lipid_ROS_ferroptosis'


def build():
    G = nx.DiGraph()
    G.graph['source'] = ('CURATED_FROM_PROPOSAL §2.2/Fig.2 (Dixon 2012; Yang 2014; '
                         'Viswanathan 2017; Warner 2000; sulfasalazine-ER 2019)')

    def N(name, kind, label=None, **kw):
        G.add_node(name, kind=kind, label=label or name.replace('_', ' '), **kw)

    # --- drugs ---
    N('sulfasalazine', 'drug', in_data=True)
    N('erastin', 'drug', in_data=True)
    N('simvastatin', 'drug', in_data=True)
    N('lovastatin', 'drug', in_data=True)
    N('RSL3', 'drug', in_data=False, note='Type II reference; absent from our matrices')
    # --- targets / processes ---
    N('SLC7A11', 'target', label='System xc-\n(SLC7A11)')
    N('HMGCR', 'target')
    N('GPX4', 'target')
    N('cystine_import', 'process', label='cystine import')
    N('cysteine', 'metabolite')
    N('GSH', 'metabolite', label='glutathione (GSH)')
    N('mevalonate_pathway', 'pathway')
    N('IPP', 'metabolite', label='IPP')
    N('seleno_tRNA', 'process', label='selenocysteine-tRNA\nmodification')
    N('GPX4_synthesis', 'process')
    N('lipid_hydroperoxides', 'process', label='lipid hydroperoxides')
    N('labile_Fe2', 'process', label='labile Fe2+\n(Fenton)')
    N(FERROPTOSIS, 'outcome', label='LIPID-ROS\nFERROPTOSIS')
    N('apoptosis', 'outcome')
    # --- core genes (proposal-named) ---
    for g, link in [('GCLC', 'GSH'), ('GSS', 'GSH'), ('ACSL4', 'lipid_hydroperoxides'),
                    ('LPCAT3', 'lipid_hydroperoxides'), ('NFE2L2', 'SLC7A11'),
                    ('FTH1', 'labile_Fe2'), ('TFRC', 'labile_Fe2'),
                    ('SLC3A2', 'SLC7A11')]:
        N(g, 'gene')
    # --- rescue / specificity controls ---
    N('ferrostatin_1', 'rescue')
    N('deferoxamine', 'rescue')
    N('NAC', 'rescue', label='N-acetylcysteine')
    N('ZVAD_FMK', 'control', label='Z-VAD-FMK')

    E = G.add_edge
    # Type I inducers
    E('sulfasalazine', 'SLC7A11', rel='inhibits')
    E('erastin', 'SLC7A11', rel='inhibits')
    E('SLC7A11', 'cystine_import', rel='sustains')
    E('cystine_import', 'cysteine', rel='produces')
    E('cysteine', 'GSH', rel='produces')
    E('GSH', 'GPX4', rel='cofactor_for')
    # Type II inducer
    E('RSL3', 'GPX4', rel='inhibits')
    # statin sensitiser route
    E('simvastatin', 'HMGCR', rel='inhibits')
    E('lovastatin', 'HMGCR', rel='inhibits')
    E('HMGCR', 'mevalonate_pathway', rel='sustains')
    E('mevalonate_pathway', 'IPP', rel='produces')
    E('IPP', 'seleno_tRNA', rel='required_for')
    E('seleno_tRNA', 'GPX4_synthesis', rel='required_for')
    E('GPX4_synthesis', 'GPX4', rel='produces')
    # GPX4 gate on survival
    E('GPX4', 'lipid_hydroperoxides', rel='clears')
    E('lipid_hydroperoxides', FERROPTOSIS, rel='drives')
    E('labile_Fe2', FERROPTOSIS, rel='drives')
    # core genes
    E('GCLC', 'GSH', rel='synthesises')
    E('GSS', 'GSH', rel='synthesises')
    E('ACSL4', 'lipid_hydroperoxides', rel='supplies_substrate')
    E('LPCAT3', 'lipid_hydroperoxides', rel='supplies_substrate')
    E('NFE2L2', 'SLC7A11', rel='upregulates')
    E('FTH1', 'labile_Fe2', rel='sequesters')
    E('TFRC', 'labile_Fe2', rel='imports_iron_for')
    E('SLC3A2', 'SLC7A11', rel='heterodimer_partner')
    # rescue logic (Figure 2)
    E('ferrostatin_1', 'lipid_hydroperoxides', rel='traps_radicals')
    E('deferoxamine', 'labile_Fe2', rel='chelates')
    E('NAC', 'cysteine', rel='restores')
    E('ZVAD_FMK', 'apoptosis', rel='inhibits')
    return G


def analyze(G):
    out = {}
    drugs = ['sulfasalazine', 'erastin', 'simvastatin', 'lovastatin', 'RSL3']
    out['shortest_to_GPX4'] = {}
    out['shortest_to_ferroptosis'] = {}
    for d in drugs:
        try:
            out['shortest_to_GPX4'][d] = nx.shortest_path(G, d, 'GPX4')
        except nx.NetworkXNoPath:
            out['shortest_to_GPX4'][d] = None
        try:
            out['shortest_to_ferroptosis'][d] = nx.shortest_path(G, d, FERROPTOSIS)
        except nx.NetworkXNoPath:
            out['shortest_to_ferroptosis'][d] = None
    deg = sorted(((n, G.in_degree(n) + G.out_degree(n)) for n in G.nodes),
                 key=lambda t: -t[1])[:8]
    out['hubs'] = [{'node': n, 'degree': int(k)} for n, k in deg]
    # proximity: sulfasalazine vs simvastatin target neighborhoods (undirected)
    U = G.to_undirected()
    try:
        out['proximity_sulfasalazine_simvastatin'] = int(
            nx.shortest_path_length(U, 'sulfasalazine', 'simvastatin'))
    except nx.NetworkXNoPath:
        out['proximity_sulfasalazine_simvastatin'] = None
    converge = set(out['shortest_to_ferroptosis'].get('sulfasalazine') or []) & \
        set(out['shortest_to_ferroptosis'].get('simvastatin') or [])
    out['convergent_nodes'] = sorted(converge - {'lipid_ROS_ferroptosis'})
    return out


def draw(G, path):
    layer = {'drug': 0, 'target': 1, 'gene': 2, 'metabolite': 3,
             'pathway': 3, 'process': 3, 'rescue': 4, 'control': 4,
             'outcome': 5}
    for n in G.nodes:
        G.nodes[n]['subset'] = layer.get(G.nodes[n].get('kind'), 3)
    pos = nx.multipartite_layout(G, subset_key='subset', align='vertical',
                                 scale=5)
    # top-to-bottom order per column, chosen so drug->target edges run short
    order = {
        0: ['RSL3', 'erastin', 'sulfasalazine', 'simvastatin', 'lovastatin'],
        1: ['GPX4', 'SLC7A11', 'HMGCR'],
        3: ['lipid_hydroperoxides', 'GPX4_synthesis', 'mevalonate_pathway',
            'seleno_tRNA', 'cysteine', 'GSH', 'IPP', 'cystine_import',
            'labile_Fe2'],
        4: ['ferrostatin_1', 'NAC', 'deferoxamine', 'ZVAD_FMK'],
        5: [FERROPTOSIS, 'apoptosis'],
    }
    by_layer = {}
    for n in G.nodes:
        by_layer.setdefault(G.nodes[n]['subset'], []).append(n)
    for layer, want in order.items():
        have = [n for n in want if n in by_layer.get(layer, [])]
        have += [n for n in by_layer.get(layer, []) if n not in have]
        ys = sorted((pos[n][1] for n in have), reverse=True)
        for n, y in zip(have, ys):
            pos[n] = (pos[n][0], y)
    colors = {'drug': '#74C0FC', 'target': '#63E6BE', 'gene': '#B2F2BB',
              'metabolite': '#FFD8A8', 'pathway': '#FFD8A8', 'process': '#E5DBFF',
              'rescue': '#FFC9C9', 'control': '#E9ECEF', 'outcome': '#FFA8A8'}
    fig, ax = plt.subplots(figsize=(16, 9))
    for k, c in colors.items():
        ns = [n for n in G.nodes if G.nodes[n].get('kind') == k]
        if ns:
            nx.draw_networkx_nodes(G, pos, nodelist=ns, node_color=c,
                                   node_size=2200, ax=ax, edgecolors='#343A40')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#868E96',
                           arrows=True, arrowsize=14, width=1.2,
                           connectionstyle='arc3,rad=0.08')
    nx.draw_networkx_labels(G, pos,
                            labels={n: G.nodes[n].get('label', n) for n in G.nodes},
                            font_size=8, ax=ax)
    ax.set_title('FerroSelect mechanism graph (curated, proposal §2.2/Fig.2)\n'
                 'left-to-right: drugs → targets → genes → processes → '
                 'rescue controls → outcome',
                 fontsize=11)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print('saved', path)


def table(path):
    rows = [
        ('sulfasalazine', 'SLC7A11 (System xc-)', 'cystine/GSH axis',
         'Type I inducer: starves cystine, GSH falls, GPX4 stalls', 'proposal [9]; PRISM data present'),
        ('erastin', 'SLC7A11 (System xc-)', 'cystine/GSH axis',
         'Type I reference inducer', 'proposal [10,11]; PRISM data present'),
        ('simvastatin', 'HMGCR', 'mevalonate -> IPP -> seleno-tRNA -> GPX4 synthesis',
         'Sensitiser (fixed sub-lethal dose): weakens GPX4 supply', 'proposal [18]; PRISM data present'),
        ('lovastatin', 'HMGCR', 'mevalonate -> IPP -> seleno-tRNA -> GPX4 synthesis',
         'Sensitiser alternative', 'PRISM data present'),
        ('RSL3', 'GPX4 (direct)', 'lipid hydroperoxide clearance',
         'Type II reference; droppable if over budget', 'proposal [3]; ABSENT from our matrices'),
        ('ferrostatin-1', 'lipid radicals', 'rescue control',
         'Must rescue if death is ferroptosis', 'required control'),
        ('deferoxamine', 'labile Fe2+', 'rescue control',
         'Must rescue if death needs iron', 'required control'),
        ('N-acetylcysteine', 'cysteine', 'rescue control',
         'Rescues Type I, not RSL3 (mechanism discriminator)', 'cheap control'),
        ('Z-VAD-FMK', 'caspases', 'specificity control',
         'Must NOT rescue (rules out apoptosis)', 'closes the argument'),
    ]
    pd.DataFrame(rows, columns=['drug', 'target', 'pathway', 'effect', 'evidence']).to_csv(path, index=False)
    print('saved', path)


def main(outdir=None):
    outdir = outdir or OUT
    G = build()
    print(f'graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
    res = analyze(G)
    json.dump(res, open(os.path.join(outdir, 'network_paths.json'), 'w'), indent=1)
    for d in ['sulfasalazine', 'simvastatin']:
        print(d, '-> ferroptosis:', ' -> '.join(res['shortest_to_ferroptosis'][d] or []))
    print('hubs:', [(h['node'], h['degree']) for h in res['hubs'][:5]])
    print('sulfasalazine~simvastatin proximity:', res['proximity_sulfasalazine_simvastatin'])
    print('convergent nodes:', res['convergent_nodes'])
    draw(G, os.path.join(outdir, 'ferroselect_network.png'))
    table(os.path.join(outdir, 'network_table.csv'))


if __name__ == '__main__':
    main()
