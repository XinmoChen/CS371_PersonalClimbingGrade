"""Figures for the design checks (run_design_checks.py), in the style of make_figures.py.
  gate_family_indoor.pdf  share of the personal gain kept against the share of climbers hurt, for the evidence gate
                          and its simpler alternatives, Protocols A and B
  paired_groups_indoor.pdf RMSE difference of PACE against the published forest, and of PACE-GBM against plain
                          boosting, by route history and by label, with 95% climber cluster bootstrap intervals"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT_DIR
FIG = os.environ.get('FIG_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#52514e', '#eda100'
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.edgecolor': '#b8b7b0', 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'lines.linewidth': 1.8,
                     'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300})
RULES = [('evidence gate', 'Evidence gate (PACE-G)', ORANGE, 'o', '-'),
         ('magnitude gate', 'Effect size only', BLUE, 's', '-'),
         ('count gate', 'Log count only', AQUA, '^', '-'),
         ('stronger shrinkage', 'Stronger shrinkage', GRAY, 'D', '--')]


def fig_gate_family():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, proto in zip(axes, ('A', 'B')):
        df = pd.read_csv(f'{OUT_DIR}/gate_family_indoor_{proto}.csv')
        for rule, lab, col, mk, ls in RULES:
            sub = df[df.rule == rule]
            if rule == 'stronger shrinkage':
                sub = sub.sort_values('param')
            else:
                sub = sub[sub.rows == 'test, equal coverage'].sort_values('coverage')
            ax.plot(100 * sub.hurt, 100 * sub.retained, color=col, marker=mk, markersize=4, linestyle=ls, label=lab)
        cov = json.load(open(f'{OUT_DIR}/design_checks_indoor.json'))[proto]['equal_coverage']['coverage']
        ev = df[(df.rule == 'evidence gate') & (np.isclose(df.param, cov))].iloc[0]
        ax.plot([100 * ev.hurt], [100 * ev.retained], marker='o', markersize=9, markerfacecolor='none', markeredgecolor=INK, linestyle='none')
        ax.annotate('$z=1.96$', (100 * ev.hurt, 100 * ev.retained), xytext=(4, -16), textcoords='offset points', fontsize=7.5, color=INK)
        ax.set_xlabel('Climbers hurt (%)')
        ax.set_title(f'Protocol {proto}', fontsize=9, color=INK2)
        ax.set_ylim(-20, 110)
    axes[0].set_ylabel('Personal gain kept (%)')
    axes[1].legend(loc='lower center', bbox_to_anchor=(-0.1, -0.5), ncol=4, fontsize=7.5)
    fig.savefig(f'{FIG}/gate_family_indoor.pdf'); fig.savefig(f'{FIG}/gate_family_indoor.png'); plt.close(fig)


def fig_paired_groups():
    df = pd.read_csv(f'{OUT_DIR}/paired_strata_indoor.csv')
    groups = ['all', 'route history 0', 'route history 1-2', 'route history 3-4', 'route history 5-9', 'route history 10+',
              'agrees with official', 'disagrees']
    labels = ['All test logs', 'Route history 0', '1 to 2', '3 to 4', '5 to 9', '10 or more', 'Agrees with official', 'Disagrees']
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, (new, ref, title) in zip(axes, (('PACE', 'B1-RF', 'PACE minus published forest'),
                                            ('PACE-GBM', 'B3-GBM', 'PACE-GBM minus plain boosting'))):
        for j, (proto, col, off) in enumerate((('A', ORANGE, -0.15), ('B', BLUE, 0.15))):
            sub = df[(df.protocol == proto) & (df.new == new) & (df.ref == ref)].set_index('group').reindex(groups)
            y = np.arange(len(groups)) + off
            ax.errorbar(1000 * sub['diff'], y, xerr=[1000 * (sub['diff'] - sub['lo']), 1000 * (sub['hi'] - sub['diff'])],
                        fmt='o', color=col, markersize=3.5, elinewidth=1.2, capsize=0, label=f'Protocol {proto}')
        ax.axvline(0, color=GRAY, linestyle='--', linewidth=1)
        ax.set_title(title, fontsize=9, color=INK2)
        ax.set_xlabel('RMSE difference ($\\times 10^{-3}$)')
        ax.set_yticks(np.arange(len(groups))); ax.set_yticklabels(labels)
    axes[0].invert_yaxis()                      # shared y axis: invert once
    axes[1].legend(loc='lower center', bbox_to_anchor=(-0.25, -0.42), ncol=2, fontsize=7.5)
    fig.savefig(f'{FIG}/paired_groups_indoor.pdf'); fig.savefig(f'{FIG}/paired_groups_indoor.png'); plt.close(fig)


if __name__ == '__main__':
    fig_gate_family()
    fig_paired_groups()
    print('saved')
