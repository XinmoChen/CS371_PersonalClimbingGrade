"""Routes whose deviation shifts during their life (Section 5, 'Routes whose grade changed').

For every route with at least 20 logs, the largest change in mean deviation between the logs before and after a
split point (at least 5 logs on each side) is found. A route is flagged when that shift is at least one grade.
The same statistic on 20 random permutations of each route's deviations gives the number of routes expected to
cross the threshold by chance. This uses each route's full history and serves only as a diagnostic of the test
error, never as a model input. Output: results/regrade_{setting}.json."""
import json, os, sys
import numpy as np, pandas as pd
from common import *

rng = np.random.default_rng(0)
MIN_LOGS, MIN_SIDE, THRESH, PERMS = 20, 5, 1.0, 20


def max_shift(d):
    n = len(d); c = np.cumsum(d); k = np.arange(MIN_SIDE, n - MIN_SIDE + 1)
    left = c[k - 1] / k; right = (c[-1] - c[k - 1]) / (n - k)
    diff = right - left; j = int(np.argmax(np.abs(diff)))
    return diff[j], int(k[j])


def rmse(y, p, mask):
    """RMSE on mask; a list of seeded predictions gives the mean RMSE over seeds, as in the main table."""
    if isinstance(p, list):
        return float(np.mean([rmse(y, q, mask) for q in p]))
    return float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))


out = {}
for setting in sys.argv[1:] or ['indoor', 'outdoor']:
    if not os.path.exists(f'{OUT_DIR}/pace_preds_{setting}.npz'):
        continue
    g = load(setting)
    rows, null = [], 0.0
    for r, grp in g.groupby('route_id', sort=False):
        if len(grp) < MIN_LOGS:
            continue
        d = grp.dev.values.astype(float)
        s, k = max_shift(d)
        rows.append((r, len(grp), s, grp.date.values[k]))
        null += np.mean([abs(max_shift(rng.permutation(d))[0]) >= THRESH for _ in range(PERMS)])
    R = pd.DataFrame(rows, columns=['route', 'n', 'shift', 'when'])
    flagged = R[R['shift'].abs() >= THRESH]
    res = {'routes_eligible': int(len(R)), 'routes_flagged': int(len(flagged)), 'expected_by_chance': round(float(null), 1),
           'shift_up': int((flagged['shift'] > 0).sum()), 'shift_down': int((flagged['shift'] < 0).sum()),
           'change_quarters': {str(k): int(v) for k, v in pd.to_datetime(flagged.when).dt.to_period('Q').value_counts().sort_index().items()},
           'share_logs_on_flagged': float(g.route_id.isin(flagged.route).mean()),
           'dis_rate_by_year': {str(k): round(float(v), 4) for k, v in (g.dev != 0).groupby(g.date.dt.year).mean().items()},
           'logs_by_year': {str(k): int(v) for k, v in g.groupby(g.date.dt.year).size().items()}}
    on = g.route_id.isin(flagged.route).values
    y = g.dev.values.astype(float)
    # is the change confined to the shifted routes? disagreement by year on other groups of logs and climbers
    yr = g.date.dt.year.values; dis = (g.dev != 0).values
    nlog = g.groupby('route_id').route_id.transform('size').values
    both = set(g.user_id[yr == 2018]) & set(g.user_id[yr == 2019]); inb = g.user_id.isin(both).values
    mon = g.date.dt.to_period('M').astype(str).values
    rate = lambda m: float(dis[m].mean()) if m.any() else None
    res['by_group'] = {k: {str(a): rate(m & (yr == a)) for a in (2017, 2018, 2019, 2020)} for k, m in
                       (('unshifted routes with 20 or more logs', (nlog >= MIN_LOGS) & ~on), ('routes with fewer than 20 logs', nlog < MIN_LOGS),
                        ('climbers who log in both 2018 and 2019', inb))}
    res['by_group']['climbers who log in both 2018 and 2019']['n_climbers'] = len(both)
    res['softer_by_year'] = {str(a): float((g.dev.values[yr == a] < 0).mean()) for a in range(2015, 2021)}
    res['harder_by_year'] = {str(a): float((g.dev.values[yr == a] > 0).mean()) for a in range(2015, 2021)}
    res['dis_rate_by_month_2018_2019'] = {m: float(dis[mon == m].mean()) for m in sorted(set(mon)) if m[:4] in ('2018', '2019')}
    for proto, suf in (('A', ''), ('B', '_B')):
        te = (split_protocol_a(g) if proto == 'A' else split_protocol_b(g, setting)) == 'test'
        P = np.load(f'{OUT_DIR}/pace_preds_{setting}{suf}.npz')
        preds = {'B0': np.zeros(len(g)), 'C': P['PACE[route+env]|pred'], 'PACE': P['PACE|pred']}
        bp = f'{OUT_DIR}/baseline_preds_{setting}.npz'
        if os.path.exists(bp):
            B = np.load(bp)
            keys = [k for k in B.files if k.startswith(f'{proto}|B3-GBM|')]
            if keys:
                preds['B3-GBM'] = [B[k].astype(float) for k in keys]
        gp = f'{OUT_DIR}/gbm_variant_preds_{setting}.npz'
        if os.path.exists(gp):
            G = np.load(gp)
            keys = [k for k in G.files if k.startswith(f'{proto}|PACE-GBM|')]
            if keys:
                preds['PACE-GBM'] = [G[k].astype(float) for k in keys]
        a, b = te & on, te & ~on
        sq = (y[te] - preds['PACE'][te]) ** 2
        res[proto] = {'test_logs': int(te.sum()), 'test_logs_on_flagged': int(a.sum()), 'share_test_on_flagged': float(a.sum() / te.sum()),
                      'share_pace_sq_error_on_flagged': float(sq[on[te]].sum() / sq.sum()),
                      'rmse_flagged': {k: rmse(y, v, a) for k, v in preds.items()},
                      'rmse_other': {k: rmse(y, v, b) for k, v in preds.items()},
                      'rmse_all': {k: rmse(y, v, te) for k, v in preds.items()}}
    out[setting] = res
    print(setting, json.dumps(res, indent=1))
    json.dump(res, open(f'{OUT_DIR}/regrade_{setting}.json', 'w'), indent=1)
