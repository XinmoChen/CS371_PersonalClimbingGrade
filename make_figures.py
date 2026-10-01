"""Paper figures, drawn from the result files. Palette: validated categorical slots (blue, orange, aqua) with
direct labels (aqua is under 3:1 contrast, so every series is also labeled); the official grade is a gray
dashed reference line, not a categorical series."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT_DIR
FIG = os.environ.get('FIG_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.edgecolor': '#b8b7b0', 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'lines.linewidth': 2,
                     'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300})
YELLOW = '#eda100'
SERIES = [('B1-RF', 'B1 SSE, random forest', BLUE, 's'), ('PACE[route+env]', 'C (no personal effect)', AQUA, '^'), ('PACE', 'PACE', ORANGE, 'o'), ('PACE-GBM', 'PACE-GBM', YELLOW, 'D')]


def fig_history(setting='indoor', proto='A'):
    df = pd.read_csv(f'{OUT_DIR}/strata_{setting}_{proto}.csv')
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, (kind, labs, xl) in zip(axes, [('route', ['0', '1-2', '3-4', '5-9', '10+'], 'Earlier logs on the route'),
                                          ('climber', ['0-4', '5-9', '10-19', '20-49', '50+'], 'Earlier logs by the climber')]):
        sub = df[df.kind == kind]
        big = set(sub[sub.n >= 200].stratum)
        labs = [l for l in labs if l in big]
        base = sub[sub.model == 'B0-official'].set_index('stratum').RMSE
        x = np.arange(len(labs))
        for m, lab, col, mk in SERIES:
            if m not in set(sub.model):
                continue
            v = sub[sub.model == m].set_index('stratum').RMSE
            gain = 100 * (1 - v.reindex(labs) / base.reindex(labs))
            ax.plot(x, gain.values, color=col, marker=mk, markersize=5, label=lab)
        ax.axhline(0, color=GRAY, linestyle='--', linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels([l.replace('-', ' to ') for l in labs])
        ax.set_xlabel(xl)
    axes[0].set_ylabel('RMSE reduction vs\nofficial grade (%)')
    axes[1].legend(loc='lower center', bbox_to_anchor=(-0.1, -0.52), ncol=4, fontsize=7.5)
    fig.savefig(f'{FIG}/history_{setting}_{proto}.pdf'); fig.savefig(f'{FIG}/history_{setting}_{proto}.png')
    plt.close(fig)


def fig_budget(setting='indoor'):
    df = pd.read_csv(f'{OUT_DIR}/budget_{setting}.csv')
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    x = np.arange(len(df))
    y = -1000 * df.gain_vs_k0.values; lo = -1000 * df.gain_hi.values; hi = -1000 * df.gain_lo.values
    ax.fill_between(x, lo, hi, color=ORANGE, alpha=0.18, linewidth=0)
    ax.plot(x, y, color=ORANGE, marker='o', markersize=5)
    ax.axhline(0, color=GRAY, linestyle='--', linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in df.k])
    ax.set_xlabel('Most recent climber logs used ($K$)'); ax.set_ylabel('RMSE reduction\n($\\times 10^{-3}$) vs $K=0$')
    fig.savefig(f'{FIG}/budget_{setting}.pdf'); fig.savefig(f'{FIG}/budget_{setting}.png'); plt.close(fig)


def fig_gate(setting='indoor', proto='A'):
    df = pd.read_csv(f'{OUT_DIR}/gate_sweep_{setting}_{proto}.csv')
    df = df[np.isfinite(df.z)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    ax = axes[0]
    ax.plot(100 * df.share_personalized_test, df.RMSE_test, color=ORANGE, marker='o', markersize=5)
    for _, r in df.iterrows():
        if r.z in (0.0, 1.0, 1.96, 3.0):
            ax.annotate(f'z={r.z:g}', (100 * r.share_personalized_test, r.RMSE_test), xytext=(4, 4), textcoords='offset points', fontsize=7.5, color=INK2)
    ax.set_xlabel('Test logs personalized (%)'); ax.set_ylabel('Test RMSE')
    ax = axes[1]
    ax.plot(100 * df.share_personalized_test, 100 * df.hurt_test, color=ORANGE, marker='o', markersize=5, label='hurt')
    ax.plot(100 * df.share_personalized_test, 100 * df.helped_test, color=BLUE, marker='s', markersize=5, label='helped')
    ax.annotate('hurt', (100 * df.share_personalized_test.iloc[0], 100 * df.hurt_test.iloc[0]), xytext=(4, 4), textcoords='offset points', fontsize=7.5, color=INK2)
    ax.annotate('helped', (100 * df.share_personalized_test.iloc[0], 100 * df.helped_test.iloc[0]), xytext=(4, -10), textcoords='offset points', fontsize=7.5, color=INK2)
    ax.set_xlabel('Test logs personalized (%)'); ax.set_ylabel('Climbers helped or hurt (%)')
    fig.tight_layout(); fig.savefig(f'{FIG}/gate_{setting}_{proto}.pdf'); fig.savefig(f'{FIG}/gate_{setting}_{proto}.png'); plt.close(fig)


def fig_errors():
    df = pd.read_csv(f'{OUT_DIR}/error_taxonomy.csv', index_col=0).sort_values('share')
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    y = np.arange(len(df)) * 1.0
    v = 100 * df.share.values
    ax.barh(y, v, color=BLUE, height=0.42)
    for yi, vi, name in zip(y, v, df.index):
        name = {'Lone dissent, no signal anywhere': 'Lone dissent, no signal on the route'}.get(name, name)
        ax.text(0, yi + 0.27, name, va='bottom', ha='left', fontsize=7.5, color=INK)
        ax.text(vi + 0.6, yi, f'{vi:.1f}%', va='center', fontsize=7.5, color=INK2)
    ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_axisbelow(True)
    ax.set_xlabel('Share of displayed errors (%)', fontsize=8); ax.tick_params(axis='x', labelsize=7.5)
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, v.max() * 1.18); ax.set_ylim(-0.45, y.max() + 0.85)
    fig.savefig(f'{FIG}/errors.pdf', bbox_inches='tight', pad_inches=0.03); fig.savefig(f'{FIG}/errors.png', bbox_inches='tight', pad_inches=0.03); plt.close(fig)


def fig_gain_grid(proto='A'):
    df = pd.read_csv(f'{OUT_DIR}/gain_grid_indoor.csv')
    df = df[df.protocol == proto]
    R = ['0', '1-2', '3-4', '5-9', '10+']; Cl = ['0-4', '5-9', '10-19', '20-49', '50+']
    Cl = [c for c in Cl if df[df.climber == c].gain.notna().any()]
    M = np.array([[df[(df.route == r) & (df.climber == c)].gain.iloc[0] for c in Cl] for r in R]) * 1000
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blue', ['#f0efec', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#104281'])
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(np.clip(M, 0, None), cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    for i in range(len(R)):
        for j in range(len(Cl)):
            v = M[i, j]
            ax.text(j, i, '' if np.isnan(v) else f'{v:.1f}'.replace('-', '\u2212'), ha='center', va='center', fontsize=7.5,
                    color='white' if (not np.isnan(v) and v > 0.6 * vmax) else INK)
    ax.set_xticks(range(len(Cl))); ax.set_xticklabels([c.replace('-', ' to ') for c in Cl], fontsize=7.5)
    ax.set_yticks(range(len(R))); ax.set_yticklabels([r.replace('-', ' to ') for r in R], fontsize=7.5)
    ax.set_xlabel('Earlier logs by the climber'); ax.set_ylabel('Earlier logs on the route'); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05); cb.ax.tick_params(labelsize=7); cb.set_label('Personal gain (RMSE $\\times 10^{-3}$)', fontsize=7.5)
    fig.savefig(f'{FIG}/gain_grid_{proto}.pdf'); fig.savefig(f'{FIG}/gain_grid_{proto}.png'); plt.close(fig)


def fig_regrade():
    """Outdoor reference shift: yearly disagreement rate and the timing of route level grade shifts."""
    R = {k: json.load(open(f'{OUT_DIR}/regrade_{k}.json')) for k in ('indoor', 'outdoor') if os.path.exists(f'{OUT_DIR}/regrade_{k}.json')}
    if 'outdoor' not in R:
        return
    o = R['outdoor']
    yrs = [y for y in sorted(o['dis_rate_by_year']) if o['logs_by_year'][y] >= 1000]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2))
    ax = axes[0]
    ax.plot(range(len(yrs)), [100 * o['dis_rate_by_year'][y] for y in yrs], color=ORANGE, marker='o', markersize=5)
    for j, y in enumerate(yrs):
        ax.text(j, 100 * o['dis_rate_by_year'][y] + 0.7, f"{100 * o['dis_rate_by_year'][y]:.1f}", ha='center', fontsize=6.5, color=INK2)
    if 'indoor' in R:
        ind = R['indoor']; tot = sum(ind['logs_by_year'].values())
        rate = sum(ind['dis_rate_by_year'][y] * ind['logs_by_year'][y] for y in ind['dis_rate_by_year']) / tot
        ax.axhline(100 * rate, color=GRAY, linestyle='--', linewidth=1)
        ax.text(3.5, 100 * rate + 0.35, 'indoor, all years', ha='center', fontsize=6.5, color=GRAY)
    ax.set_xticks(range(len(yrs))); ax.set_xticklabels(yrs, fontsize=7)
    ax.set_ylabel('Outdoor logs that\ndisagree (%)'); ax.set_ylim(0, None)
    ax = axes[1]
    q = pd.Series(o['change_quarters']); q.index = pd.PeriodIndex(q.index, freq='Q')
    full = pd.period_range(q.index.min(), q.index.max(), freq='Q'); q = q.reindex(full, fill_value=0)
    ax.bar(range(len(q)), q.values, color=BLUE, width=0.7)
    ticks = [j for j, p in enumerate(q.index) if p.quarter == 1]
    ax.set_xticks(ticks); ax.set_xticklabels([str(q.index[j].year) for j in ticks], fontsize=7)
    ax.set_xlabel('Quarter of the shift')
    ax.set_ylabel('Outdoor routes that shift\nby a grade or more')
    ax.grid(axis='x', visible=False); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f'{FIG}/regrade.pdf', bbox_inches='tight', pad_inches=0.03); fig.savefig(f'{FIG}/regrade.png', bbox_inches='tight', pad_inches=0.03); plt.close(fig)


def fig_development():
    """Left: prequential development RMSE by route penalty. Right: next log after n unanimous earlier logs."""
    t = pd.read_csv(f'{OUT_DIR}/pace_tuning_all_indoor.csv')
    if 'climber_grade' in t:
        t = t[t.climber_grade.isna()]
    c = pd.read_csv(f'{OUT_DIR}/consensus_shape.csv')
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
    for ax in axes: ax.tick_params(labelsize=6.5)
    ax = axes[0]
    for proto, lu, col, mk in (('A', 40, BLUE, 'o'), ('B', 20, ORANGE, 's')):
        sub = t[(t.climber == lu) & (t.env == 30)].sort_values('route')
        ax.plot(sub.route, sub[f'devRMSE_{proto}'], color=col, marker=mk, markersize=3, linewidth=1.4, label=f'Protocol {proto}')
        b = sub.loc[sub[f'devRMSE_{proto}'].idxmin()]
        ax.plot([b.route], [b[f'devRMSE_{proto}']], marker=mk, markersize=6, markerfacecolor='none', markeredgecolor=INK, linestyle='none')
    ax.set_xticks([1, 2, 3, 4, 5, 6, 8]); ax.set_xlabel('Route penalty $\\lambda_c$', fontsize=7); ax.set_ylabel('Development RMSE', fontsize=7); ax.legend(fontsize=5.5)
    ax = axes[1]
    u = c[c.unanimous & (c.S.abs() == c.n)]
    for sign, col, mk, lab in ((1, ORANGE, 'o', 'all +1'), (-1, BLUE, 's', 'all $-1$ (sign flipped)')):
        sub = u[np.sign(u.S) == sign].sort_values('n')
        ax.plot(sub.n, sign * sub['mean'], color=col, marker=mk, markersize=3, linewidth=1.4, label=lab)
    lin = u[u.S > 0].sort_values('n')
    ax.plot(lin.n, lin.linear, color=GRAY, linestyle='--', linewidth=1.0, label='linear pooling')
    ax.set_xlabel('Earlier logs ($n$)', fontsize=7); ax.set_ylabel('Next log deviation', fontsize=7)
    ax.set_xticks(sorted(u.n.unique())); ax.legend(fontsize=5.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/development.pdf', bbox_inches='tight', pad_inches=0.03); fig.savefig(f'{FIG}/development.png', bbox_inches='tight', pad_inches=0.03); plt.close(fig)


if __name__ == '__main__':
    todo = sys.argv[1:] or ['history', 'budget', 'gate', 'errors', 'gain_grid', 'regrade', 'development']
    if 'history' in todo: fig_history('indoor', 'A'); fig_history('indoor', 'B')
    if 'budget' in todo: fig_budget('indoor')
    if 'gate' in todo: fig_gate('indoor', 'A')
    if 'errors' in todo: fig_errors()
    if 'gain_grid' in todo: fig_gain_grid('A'); fig_gain_grid('B')
    if 'regrade' in todo: fig_regrade()
    if 'development' in todo: fig_development()
    print('figures in', FIG)
