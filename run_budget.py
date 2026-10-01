"""RQ2: how much climber history does the personal effect need?
For every Protocol A test log, the personal effect is recomputed from only the climber's k most recent earlier
logs (k in 0, 1, 2, 5, 10, 20, 50, all), holding the route, setter and fixed parts at the values PACE used.
Residuals of those k logs are taken against the day-start PACE snapshot of the prediction day, so the
recomputation uses exactly the information PACE had, minus the older climber logs."""
import sys, json
import numpy as np, pandas as pd
from common import *
from metrics import rmse, rounded, climber_bootstrap

setting = sys.argv[1] if len(sys.argv) > 1 else 'indoor'
g = load(setting)
d = g.dev.values.astype(float)
X = np.column_stack([np.ones(len(g)), g.grade_id.values - 13.0])
P = np.load(f'{OUT_DIR}/pace_preds_{setting}.npz')
S = np.load(f'{OUT_DIR}/pace_snapshots_{setting}.npz')
LAM = json.load(open(f'{OUT_DIR}/pace_lambdas_{setting}.json'))
env_key = g.route_setter_id.values if setting == 'indoor' else pd.to_numeric(g.rock_type, errors='coerce').fillna(-1).values
cu, cr, ce = codes(g.user_id.values), codes(g.route_id.values), codes(env_key)
periods = S['periods']; per_idx = np.searchsorted(periods, g.day.values)   # period = day for indoor runs
assert (periods[per_idx] == g.day.values).all()
beta, effR, effE = S['beta'], S['eff_route'], S['eff_env']
sp = split_protocol_a(g); te = np.where(sp == 'test')[0]
base = P['PACE|pred'] - P['PACE|val|climber']            # PACE without its personal effect (same other terms)
KS = [0, 1, 2, 5, 10, 20, 50, 10 ** 6]
out = {k: np.full(len(g), np.nan) for k in KS}
t = g.t.values
for u, rows in pd.Series(np.arange(len(g))).groupby(cu):
    rows = rows.values                                    # this climber's logs in time order
    for n in rows[np.isin(rows, te)]:
        prior = rows[t[rows] < t[n]]
        j = per_idx[n]
        resid = d[prior] - X[prior] @ beta[j] - effR[j][cr[prior]] - effE[j][ce[prior]]
        for k in KS:
            r = resid[-k:] if k > 0 else resid[:0]
            uk = r.sum() / (len(r) + LAM['climber'])
            out[k][n] = base[n] + uk
rows = []
yt = d[te]; users = g.user_id.values[te]
full = P['PACE|pred'][te]
for k in KS:
    p = out[k][te]
    bs = climber_bootstrap(yt, {'k': p, 'base': base[te]}, users, B=1000, ref='base')
    rows.append(dict(k=('all' if k == 10 ** 6 else k), RMSE=rmse(yt, p), RMSE_r=rmse(yt, rounded(p)),
                     RMSE_dis=rmse(yt[yt != 0], p[yt != 0]), gain_vs_k0=bs['k']['diff'],
                     gain_lo=bs['k']['diff_lo'], gain_hi=bs['k']['diff_hi'],
                     max_abs_diff_from_PACE=float(np.nanmax(np.abs(p - full))) if k == 10 ** 6 else np.nan))
df = pd.DataFrame(rows)
print(df.round(5).to_string())
df.to_csv(f'{OUT_DIR}/budget_{setting}.csv', index=False)
# the latest 50 logs against the full history, paired on the same climber resamples
bs50 = climber_bootstrap(yt, {'k50': out[50][te], 'all': out[10 ** 6][te]}, users, B=1000, ref='all')['k50']
json.dump(bs50, open(f'{OUT_DIR}/budget_k50_vs_all_{setting}.json', 'w'), indent=1)
print('k=50 vs all', bs50)
np.savez_compressed(f'{OUT_DIR}/budget_preds_{setting}.npz', **{str(k): v[te] for k, v in out.items()}, te=te)
