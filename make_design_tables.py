"""LaTeX tables for the design checks (run_design_checks.py).
  tables/gate_family_indoor.tex    the gate and its simpler alternatives at the gate's coverage, Protocols A and B
  tables/gate_evidence_indoor.tex  share of test logs the gate personalizes by climber history, observed and
                                   predicted under the Gaussian model"""
import os, json
import numpy as np, pandas as pd
from common import OUT_DIR
TAB = os.environ.get('TAB_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables')
os.makedirs(TAB, exist_ok=True)
D = json.load(open(f'{OUT_DIR}/design_checks_indoor.json'))
pct = lambda x: f'{100 * x:.1f}'


def gate_family():
    rows = [('evidence gate', 'Evidence gate, $z=1.96$'), ('magnitude gate', 'Effect size only'),
            ('count gate', 'Log count only'), ('track record gate', 'Own track record'), ('stronger shrinkage', 'Stronger shrinkage')]
    cells = {}
    for proto in ('A', 'B'):
        fam = pd.read_csv(f'{OUT_DIR}/gate_family_indoor_{proto}.csv')
        eq = D[proto]['equal_coverage']; cov = eq['coverage']; lu = eq['shrinkage_lambda_matched_on_hurt']
        for key, _ in rows:
            if key == 'stronger shrinkage':
                r = fam[(fam.rule == key) & np.isclose(fam.param, lu)].iloc[0]
            else:
                r = fam[(fam.rule == key) & np.isclose(fam.param, cov)].iloc[0]
            cells[(proto, key)] = r
    lines = [r'\begin{tabular}{lcccccc}', r'\toprule',
             r' & \multicolumn{3}{c}{\textbf{Protocol A}} & \multicolumn{3}{c}{\textbf{Protocol B}} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
             r'\textbf{Rule} & Kept & Helped & Hurt & Kept & Helped & Hurt \\', r'\midrule']
    for key, lab in rows:
        if key == 'stronger shrinkage':
            lab += f' ($\\lambda_u={int(D["A"]["equal_coverage"]["shrinkage_lambda_matched_on_hurt"])}$)'
        a, b = cells[('A', key)], cells[('B', key)]
        lines.append(f'{lab} & {pct(a.retained)} & {pct(a.helped)} & {pct(a.hurt)} & {pct(b.retained)} & {pct(b.helped)} & {pct(b.hurt)} \\\\')
    # each rule with its threshold tuned on development rows, as the gate's z is
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{7}{l}{\textit{Each threshold tuned on development rows}} \\')
    fmt = {'evidence gate': lambda v: f'$z={v:g}$', 'magnitude gate': lambda v: f'$t={v:g}$', 'count gate': lambda v: f'$K={int(v)}$',
           'track record gate': lambda v: f'$r={v:g}$', 'stronger shrinkage': lambda v: f'$\\lambda_u={int(v)}$'}
    names = {'evidence gate': 'Evidence gate', 'magnitude gate': 'Effect size only', 'count gate': 'Log count only',
             'track record gate': 'Own track record', 'stronger shrinkage': 'Shrinkage'}
    for key, _ in rows:
        ta, tb = D['A']['tuned_on_dev'][key], D['B']['tuned_on_dev'][key]
        par = fmt[key](ta['param']) if ta['param'] == tb['param'] else fmt[key](ta['param']) + ', ' + fmt[key](tb['param'])
        lines.append(f'{names[key]} ({par}) & {pct(ta["retained"])} & {pct(ta["helped"])} & {pct(ta["hurt"])} & '
                     f'{pct(tb["retained"])} & {pct(tb["helped"])} & {pct(tb["hurt"])} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(f'{TAB}/gate_family_indoor.tex', 'w').write('\n'.join(lines) + '\n')


def gate_evidence():
    order = ['[0, 10)', '[10, 20)', '[20, 50)', '[50, 100)', '[100, 200)', '[200, 400)', '[400, 1000000000)']
    labels = ['0 to 9', '10 to 19', '20 to 49', '50 to 99', '100 to 199', '200 to 399', '400 or more']
    by = {p: {r['bin']: r for r in D[p]['gate_evidence']['by_history']} for p in ('A', 'B')}
    lines = [r'\begin{tabular}{lrccrcc}', r'\toprule',
             r' & \multicolumn{3}{c}{\textbf{Protocol A}} & \multicolumn{3}{c}{\textbf{Protocol B}} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
             r'\textbf{Climber history} & Logs & Obs. & Pred. & Logs & Obs. & Pred. \\', r'\midrule']
    for b, lab in zip(order, labels):
        cells = []
        for p in ('A', 'B'):
            r = by[p].get(b)
            cells += [f'{r["logs"]:,}', pct(r['observed']), pct(r['predicted'])] if r else ['0', '', '']
        lines.append(f'{lab} & ' + ' & '.join(cells) + r' \\')
    oa, ob = D['A']['gate_evidence'], D['B']['gate_evidence']
    lines.append(r'\midrule')
    lines.append(f'All & & {pct(oa["overall_observed"])} & {pct(oa["overall_predicted"])} & & {pct(ob["overall_observed"])} & {pct(ob["overall_predicted"])} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(f'{TAB}/gate_evidence_indoor.tex', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    gate_family()
    gate_evidence()
    print('saved')
