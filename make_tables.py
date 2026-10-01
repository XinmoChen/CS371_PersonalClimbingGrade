"""Writes every LaTeX table of the paper directly from the result files, so no number is typed by hand."""
import json, os
import numpy as np, pandas as pd
from common import OUT_DIR
TAB = os.environ.get('TAB_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables')
os.makedirs(TAB, exist_ok=True)
f3 = lambda x: '' if pd.isna(x) else f'{x:.3f}'
f4 = lambda x: '' if pd.isna(x) else f'{x:.4f}'
pct = lambda x: '' if pd.isna(x) else f'{100 * x:.1f}'
NAMES = {'B0-official': 'B0 Official grade', 'B1-LR': 'B1 SSE, linear reg.', 'B1-RF': 'B1 SSE, random forest',
         'B2-SVD': 'B2 SVD', 'B3-GBM': 'B3 Gradient boosting', 'PACE[route+env]': 'C Community and context',
         'PACE': 'PACE', 'PACE-G': 'PACE-G', 'PACE-X': 'PACE-X', 'PACE-GBM': 'PACE-GBM'}
ORDER = list(NAMES)
PUBLISHED = {'indoor': {'B0-official': (0.418, 0.239), 'B1-LR': (0.381, 0.219), 'B1-RF': (0.378, 0.213), 'B2-SVD': (0.384, 0.222)},
             'outdoor': {'B0-official': (0.339, 0.191), 'B1-LR': (0.317, 0.176), 'B1-RF': (0.321, 0.178), 'B2-SVD': (0.322, 0.174)}}


def pm(r, col, fmt=f4):
    v = fmt(r[col]); sd = r.get(col + '_sd', 0.0)
    if r.get('seeds', 1) <= 1 or pd.isna(sd):
        return v
    return v + (f'\\,{{\\scriptsize$\\pm${sd:.4f}}}' if sd >= 5e-5 else '\\,{\\scriptsize$\\pm$$<$0.0001}')


def splits():
    df = pd.read_csv(f'{OUT_DIR}/split_stats.csv')
    lines = ['\\begin{tabular}{llrrrrrr}', '\\toprule',
             ' & & \\textbf{Logs} & \\textbf{Clm.} & \\textbf{Routes} & \\textbf{Dis.} & \\textbf{New} & \\textbf{Med. hist.} \\\\', '\\midrule']
    for (setting, proto), grp in df.groupby(['setting', 'protocol'], sort=False):
        for i, r in enumerate(grp.itertuples()):
            lab = f'{setting.capitalize()} {proto}' if i == 0 else ''
            lines.append(f'{lab} & {r.split} & {r.logs:,} & {r.climbers:,} & {r.routes:,} & {pct(r.disagree)} & {pct(r.new_route)} & {int(r.climber_hist_median)} \\\\')
        lines.append('\\midrule' if not (setting == 'outdoor' and proto == 'B') else '\\bottomrule')
    lines.append('\\end{tabular}')
    open(f'{TAB}/splits.tex', 'w').write('\n'.join(lines) + '\n')


def main_table(setting):
    A = pd.read_csv(f'{OUT_DIR}/main_{setting}_A.csv').set_index('model')
    B = pd.read_csv(f'{OUT_DIR}/main_{setting}_B.csv').set_index('model')
    boot = json.load(open(f'{OUT_DIR}/boot_{setting}_A.json'))
    ref = [k for k in boot if k.startswith('ref_B1')][0]
    lines = ['\\begin{tabular}{l cccccc cccc}', '\\toprule',
             ' & \\multicolumn{6}{c}{\\textbf{Protocol A (published split)}} & \\multicolumn{4}{c}{\\textbf{Protocol B (forward split)}} \\\\',
             '\\cmidrule(lr){2-7}\\cmidrule(lr){8-11}',
             '\\textbf{Model} & RMSE & RMSE$_r$ & uRMSE & RMSE$_{\\mathrm{dis}}$ & F1 & AP$_{\\mathrm{dis}}$ & RMSE & RMSE$_r$ & RMSE$_{\\mathrm{dis}}$ & F1 \\\\', '\\midrule']
    pub = PUBLISHED[setting]
    lines.append('\\multicolumn{11}{l}{\\textit{Published results of \\citet{andric2021}}} \\\\')
    for k, (r, u) in pub.items():
        lines.append(f'{NAMES[k]} & & {r:.3f} & {u:.3f} & & & & & & & \\\\')
    lines.append('\\midrule')
    lines.append('\\multicolumn{11}{l}{\\textit{This paper}} \\\\')
    # PACE-X is tuned indoors only (its interaction penalties were not searched outdoors), so it is not listed there
    shown = [m for m in ORDER if m in A.index and not (setting == 'outdoor' and m == 'PACE-X')]
    for m in shown:
        a = A.loc[m].to_dict(); a['seeds'] = A.loc[m, 'seeds']
        bestA = {c: A.loc[shown, c].agg('max' if c in ('macroF1', 'AP_dis') else 'min') for c in ('RMSE', 'RMSE_r', 'userRMSE_r', 'RMSE_dis', 'macroF1', 'AP_dis')}
        PR = {'RMSE': 4, 'RMSE_r': 4, 'userRMSE_r': 4, 'RMSE_dis': 3, 'macroF1': 3, 'AP_dis': 3}
        bb = lambda c, txt: ('\\textbf{' + txt + '}') if round(a[c], PR[c]) == round(bestA[c], PR[c]) else txt
        cells = [bb('RMSE', pm(a, 'RMSE')), bb('RMSE_r', pm(a, 'RMSE_r')), bb('userRMSE_r', f4(a['userRMSE_r'])), bb('RMSE_dis', pm(a, 'RMSE_dis', f3)), bb('macroF1', f3(a['macroF1'])), bb('AP_dis', f3(a['AP_dis']))]
        if m in B.index:
            b = B.loc[m].to_dict(); b['seeds'] = B.loc[m, 'seeds']
            bestB = {c: B.loc[[x for x in shown if x in B.index], c].agg('max' if c == 'macroF1' else 'min') for c in ('RMSE', 'RMSE_r', 'RMSE_dis', 'macroF1')}
            bq = lambda c, txt: ('\\textbf{' + txt + '}') if round(b[c], PR[c]) == round(bestB[c], PR[c]) else txt
            cells += [bq('RMSE', pm(b, 'RMSE')), bq('RMSE_r', pm(b, 'RMSE_r')), bq('RMSE_dis', pm(b, 'RMSE_dis', f3)), bq('macroF1', f3(b['macroF1']))]
        else:
            cells += ['', '', '', '']
        if m == 'PACE[route+env]':
            lines.append('\\midrule')
        lines.append(f'{NAMES[m]} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    open(f'{TAB}/main_{setting}.tex', 'w').write('\n'.join(l for l in lines if l) + '\n')


def strata(setting, proto='A'):
    df = pd.read_csv(f'{OUT_DIR}/strata_{setting}_{proto}.csv')
    models = ['B0-official', 'B1-RF', 'B3-GBM', 'PACE[route+env]', 'PACE', 'PACE-G', 'PACE-GBM']
    out = []
    for kind, labs in (('route', ['0', '1-2', '3-4', '5-9', '10+']), ('climber', ['0-4', '5-9', '10-19', '20-49', '50+'])):
        sub = df[df.kind == kind]
        labs = [l for l in labs if l in set(sub.stratum)]
        head = ' & '.join(l.replace('-', ' to ').replace('+', '+') for l in labs)
        lines = ['\\begin{tabular}{l' + 'c' * len(labs) + '}', '\\toprule', f'\\textbf{{{kind.capitalize()} history}} & ' + head + ' \\\\',
                 'Share of test logs & ' + ' & '.join(pct(sub[(sub.stratum == l)].n.iloc[0] / sub.drop_duplicates('stratum').n.sum()) for l in labs) + ' \\\\',
                 'Disagreement rate & ' + ' & '.join(pct(sub[(sub.stratum == l)].dis_rate.iloc[0]) for l in labs) + ' \\\\', '\\midrule']
        for m in models:
            vals = []
            col = sub[sub.model == m].set_index('stratum')
            if col.empty:
                continue
            for l in labs:
                v = col.loc[l, 'RMSE'] if l in col.index else np.nan
                bestv = sub[(sub.stratum == l) & sub.model.isin(models)].RMSE.min()
                vals.append(('\\textbf{' + f3(v) + '}') if abs(v - bestv) < 1e-12 else f3(v))
            lines.append(f'{NAMES[m]} & ' + ' & '.join(vals) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}']
        open(f'{TAB}/strata_{kind}_{setting}_{proto}.tex', 'w').write('\n'.join(lines) + '\n')


def attribution(setting='indoor'):
    rows = []
    # outdoors under Protocol B both models trail the official grade, so shares of a negative gain are not defined
    for proto in (('A', 'B') if setting == 'indoor' else ('A',)):
        for kind, fn in (('PACE', f'shapley_{setting}_{proto}.csv'), ('SSE', f'shapley_sse_{setting}_{proto}.csv')):
            p = f'{OUT_DIR}/{fn}'
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p).set_index('source')
            tot = df.loc[[i for i in df.index if not i.startswith('subset')], 'mse_reduction'].sum()   # full gain over the official grade, grade term included
            rows.append((proto, kind, df, tot))
    lines = ['\\begin{tabular}{llcccc}', '\\toprule',
             '\\textbf{Split} & \\textbf{Model} & \\textbf{Personal} & \\textbf{Community} & \\textbf{Context} & \\textbf{Total} \\\\', '\\midrule']
    for proto, kind, df, tot in rows:
        ctx = 'env' if 'env' in df.index else ('setter' if 'setter' in df.index else None)
        cells = []
        for src in ('climber', 'route', ctx):
            if src is None or src not in df.index:
                cells.append('')
            else:
                v = df.loc[src, 'mse_reduction']
                cells.append(f'{1000 * v:.2f} ({100 * v / tot:.0f}\\%)')
        lines.append(f'{proto} & {kind} & ' + ' & '.join(cells) + f' & {1000 * tot:.2f} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    open(f'{TAB}/attribution_{setting}.tex', 'w').write('\n'.join(lines) + '\n')


def helped(setting='indoor'):
    lab = [('PACE', 'PACE-noU', 'PACE, every climber'), ('PACE-G', 'PACE-noU', 'PACE-G, gated'),
           ('B1-LR', None, 'B1 SSE, linear reg.'), ('B3-GBM', None, 'B3 Gradient boosting'), ('PACE-GBM', None, 'PACE-GBM')]
    lines = ['\\begin{tabular}{llcccc}', '\\toprule', ' & & \\multicolumn{2}{c}{\\textbf{Continuous}} & \\multicolumn{2}{c}{\\textbf{Displayed}} \\\\',
             '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}', '\\textbf{Split} & \\textbf{Personal term in} & Helped & Hurt & Helped & Hurt \\\\', '\\midrule']
    for proto in ('A', 'B'):
        df = pd.read_csv(f'{OUT_DIR}/helped_{setting}_{proto}.csv')
        df = df[df.tol == 1e-3]
        for new, ref, name in lab:
            sub = df[df.new == new]
            if ref is not None:
                sub = sub[sub.reference == ref]
            c = sub[sub.output == 'continuous'].iloc[0]; dsp = sub[sub.output == 'displayed'].iloc[0]
            lines.append(f'{proto} & {name} & {pct(c.helped)} & {pct(c.hurt)} & {pct(dsp.helped)} & {pct(dsp.hurt)} \\\\')
        lines.append('\\midrule' if proto == 'A' else '\\bottomrule')
    lines.append('\\end{tabular}')
    open(f'{TAB}/helped_{setting}.tex', 'w').write('\n'.join(lines) + '\n')


def ablation(setting='indoor'):
    A = pd.read_csv(f'{OUT_DIR}/main_{setting}_A.csv').set_index('model')
    B = pd.read_csv(f'{OUT_DIR}/main_{setting}_B.csv').set_index('model')
    items = [('PACE', 'PACE (full)'), ('PACE-sep', 'Separate residual means'), ('PACE-k4', 'Penalties fixed at $k=4$'),
             ('PACE-day', 'No same day update'), ('PACE-static', 'One static fit'), ('PACE-X', 'PACE-X (grade interactions)')]
    lines = ['\\begin{tabular}{lcccc}', '\\toprule', ' & \\multicolumn{2}{c}{\\textbf{Protocol A}} & \\multicolumn{2}{c}{\\textbf{Protocol B}} \\\\',
             '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}', '\\textbf{Variant} & RMSE & RMSE$_{\\mathrm{dis}}$ & RMSE & RMSE$_{\\mathrm{dis}}$ \\\\', '\\midrule']
    for m, lab in items:
        ca = [f4(A.loc[m, 'RMSE']), f3(A.loc[m, 'RMSE_dis'])] if m in A.index else ['', '']
        cb = [f4(B.loc[m, 'RMSE']), f3(B.loc[m, 'RMSE_dis'])] if m in B.index else ['', '']
        lines.append(f'{lab} & ' + ' & '.join(ca + cb) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    open(f'{TAB}/ablation_{setting}.tex', 'w').write('\n'.join(lines) + '\n')


def budget(setting='indoor'):
    df = pd.read_csv(f'{OUT_DIR}/budget_{setting}.csv')
    lines = ['\\begin{tabular}{l' + 'c' * len(df) + '}', '\\toprule', '$K$ & ' + ' & '.join(str(k) for k in df.k) + ' \\\\', '\\midrule',
             'RMSE & ' + ' & '.join(f4(v) for v in df.RMSE) + ' \\\\',
             'RMSE$_{\\mathrm{dis}}$ & ' + ' & '.join(f3(v) for v in df.RMSE_dis) + ' \\\\', '\\bottomrule', '\\end{tabular}']
    open(f'{TAB}/budget_{setting}.tex', 'w').write('\n'.join(lines) + '\n')


def ordinal(setting='indoor', proto='A'):
    df = pd.read_csv(f'{OUT_DIR}/ordinal_{setting}_{proto}.csv')
    lab = {'PACE': 'PACE', 'PACE-G': 'PACE-G', 'PACE-X': 'PACE-X', 'B1-RF': 'B1 SSE, random forest', 'B3-GBM': 'B3 Gradient boosting', 'class prior': 'Class frequencies'}
    link = {'ordinal': 'Ordinal', 'gaussian': 'Gaussian'}
    lines = ['\\begin{tabular}{llccccc}', '\\toprule', '\\textbf{Scores} & \\textbf{Link} & Log loss & Brier & ECE & AP$_{\\mathrm{hard}}$ & AP$_{\\mathrm{soft}}$ \\\\', '\\midrule']
    for _, r in df.iterrows():
        lines.append(f'{lab.get(r.model, r.model)} & {link.get(r.link, "")} & {f3(r.logloss)} & {f3(r.brier)} & {f3(r.ece)} & {f3(r.get("AP_hard", np.nan))} & {f3(r.get("AP_soft", np.nan))} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    open(f'{TAB}/ordinal_{setting}_{proto}.tex', 'w').write('\n'.join(lines) + '\n')


def transfer():
    df = pd.read_csv(f'{OUT_DIR}/transfer.csv')
    lab = {'own': 'same setting', 'none': 'none', 'other': 'other setting only', 'other, scaled': 'other setting, scaled',
           'pooled': 'both settings pooled'}
    lines = ['\\begin{tabular}{llcccc}', '\\toprule', '\\textbf{Predicting} & \\textbf{Personal effect from} & RMSE & Change & 95\\% interval & RMSE$_{\\mathrm{dis}}$ \\\\', '\\midrule']
    for tgt, grp in df.groupby('target', sort=False):
        for i, r in enumerate(grp.itertuples()):
            first = f'{tgt.capitalize()} ({r.n_climbers:,})' if i == 0 else ''
            ch = '' if r.effect == 'none' else f'${r.diff_vs_none:+.4f}$'
            iv = '' if r.effect == 'none' else f'$[{r.diff_lo:+.4f}, {r.diff_hi:+.4f}]$'
            lines.append(f'{first} & {lab[r.effect]} & {f4(r.RMSE)} & {ch} & {iv} & {f3(r.RMSE_dis)} \\\\')
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'; lines.append('\\end{tabular}')
    open(f'{TAB}/transfer.tex', 'w').write('\n'.join(lines) + '\n')


def bylabel(setting='indoor'):
    """Precision and recall on softer and harder logs, macro F1 and average precision for disagreement."""
    items = [('B1-LR', 'B1 SSE, linear reg.'), ('B1-RF', 'B1 SSE, random forest'), ('B2-SVD', 'B2 SVD'), ('B3-GBM', 'B3 Gradient boosting'),
             ('PACE', 'PACE'), ('PACE-G', 'PACE-G'), ('PACE-X', 'PACE-X'), ('PACE-GBM', 'PACE-GBM')]
    lines = ['\\begin{tabular}{llcccccc}', '\\toprule', ' & & \\multicolumn{2}{c}{\\textbf{Softer}} & \\multicolumn{2}{c}{\\textbf{Harder}} & & \\\\',
             '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}', '\\textbf{Split} & \\textbf{Model} & Prec. & Rec. & Prec. & Rec. & Macro F1 & AP$_{\\mathrm{dis}}$ \\\\', '\\midrule']
    for proto in ('A', 'B'):
        df = pd.read_csv(f'{OUT_DIR}/main_{setting}_{proto}.csv').set_index('model')
        for m, lab in items:
            if m not in df.index or (setting == 'outdoor' and m == 'PACE-X'):
                continue
            r = df.loc[m]
            lines.append(f'{proto} & {lab} & {f3(r.P_soft)} & {f3(r.R_soft)} & {f3(r.P_hard)} & {f3(r.R_hard)} & {f3(r.macroF1)} & {f3(r.AP_dis)} \\\\')
        lines.append('\\midrule' if proto == 'A' else '\\bottomrule')
    lines.append('\\end{tabular}')
    open(f'{TAB}/bylabel_{setting}.tex', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    import sys
    splits()
    for s in sys.argv[1:] or ['indoor']:
        for fn, args in ((main_table, (s,)), (strata, (s, 'A')), (strata, (s, 'B')), (attribution, (s,)), (helped, (s,)),
                         (ablation, (s,)), (budget, (s,)), (ordinal, (s, 'A')), (bylabel, (s,))):
            try:
                fn(*args)
            except FileNotFoundError as e:
                print('skipped', fn.__name__, args, '(missing', e.filename, ')')
    if os.path.exists(f'{OUT_DIR}/transfer.csv'):
        transfer()
    print('tables written to', TAB)
