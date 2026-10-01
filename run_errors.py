"""Error analysis (Section 5). Population: Protocol A indoor test logs whose displayed PACE prediction is wrong
on a disagreement log (the climber's grade differs from the official one and PACE displays another value).
Each error gets one cause by explicit rules applied in priority order; the rules were written after reading
a random sample of errors and are then checked against a fresh random sample of 100 errors (errors_sample.csv).
Later logs on the route are used ONLY to diagnose errors, never to predict."""
import json
import numpy as np, pandas as pd
from common import *
from metrics import rounded
from pace_model import gated

g = add_history_counts(load('indoor'))
P = np.load(f'{OUT_DIR}/pace_preds_indoor.npz'); LAM = json.load(open(f'{OUT_DIR}/pace_lambdas_indoor.json'))
d = g.dev.values.astype(float); p = P['PACE|pred']; u = P['PACE|val|climber']; nu = P['PACE|cnt|climber']
_, keep = gated({'pred': p, 'block_value': {'climber': u}, 'block_count': {'climber': nu}, 'sigma2': P['PACE|sigma2']}, 'climber', LAM['climber'], 1.96)
sp = split_protocol_a(g)
t = g.t.values; r = g.route_id.values; cl = g.user_id.values
sgn = np.sign(d)
# earlier and later same-direction counts on the route
same_dir_prior = np.zeros(len(g)); same_dir_later = np.zeros(len(g)); n_later = np.zeros(len(g))
for s in (-1, 1):
    m = (sgn == s).astype(float)
    sp_, n_ = prior_sum_count(r, t, m)
    tot = pd.Series(m).groupby(r).transform('sum').values
    cnt_tot = pd.Series(np.ones(len(g))).groupby(r).transform('sum').values
    # later = total - prior - same timestamp group (incl. self); approximate by total - prior - self
    later_s = tot - sp_ - m
    sel = sgn == s
    same_dir_prior[sel] = sp_[sel]; same_dir_later[sel] = later_s[sel]
n_later = pd.Series(np.ones(len(g))).groupby(r).transform('sum').values - g.n_route.values - 1
cl_same_prior = np.zeros(len(g))
for s in (-1, 1):
    m = (sgn == s).astype(float)
    sp_, n_ = prior_sum_count(cl, t, m)
    cl_same_prior[sgn == s] = (sp_ / np.maximum(n_, 1))[sgn == s]
route_mean_prior, _ = prior_sum_count(r, t, d)
route_mean_prior = route_mean_prior / np.maximum(g.n_route.values, 1)

te = sp == 'test'
err = te & (d != 0) & (rounded(p) != d)
idx = np.where(err)[0]
cause = np.full(len(g), '', dtype=object)
disp = rounded(p)
for n in idx:
    s = sgn[n]
    if g.n_route.values[n] == 0:
        c = 'New route, no community evidence'
    elif same_dir_prior[n] == 0 and same_dir_later[n] == 0 and cl_same_prior[n] < 0.10:
        c = 'Lone dissent, no signal anywhere'
    elif same_dir_prior[n] == 0 and n_later[n] > 0 and same_dir_later[n] / max(n_later[n], 1) >= 0.25:
        c = 'Early dissent, confirmed later'
    elif (np.sign(route_mean_prior[n]) == -s) or (keep[n] and np.sign(u[n]) == -s and abs(u[n]) >= 0.1):
        c = 'Evidence pointed the other way'
    elif np.sign(p[n]) == s and disp[n] == 0:
        c = 'Right direction, too cautious to show'
    elif np.sign(disp[n]) == s and abs(disp[n]) < abs(d[n]):
        c = 'Right direction, too few steps'
    else:
        c = 'Other'
    cause[n] = c
df = pd.DataFrame({'row': idx, 'user_id': cl[idx], 'route_id': r[idx], 'setter': g.route_setter_id.values[idx],
                   'date': g.date.values[idx], 'grade_id': g.grade_id.values[idx], 'dev': d[idx], 'pred': p[idx],
                   'pred_display': rounded(p[idx]), 'u': u[idx], 'u_kept': keep[idx], 'n_climber': g.n_climber.values[idx],
                   'n_route': g.n_route.values[idx], 'route_prior_mean': route_mean_prior[idx],
                   'route_same_dir_prior': same_dir_prior[idx], 'route_same_dir_later': same_dir_later[idx],
                   'route_n_later': n_later[idx], 'climber_same_dir_rate': cl_same_prior[idx], 'cause': cause[idx]})
dist = df.cause.value_counts(normalize=True).rename('share').to_frame()
dist['count'] = df.cause.value_counts()
dist['mean_abs_dev'] = df.groupby('cause').dev.apply(lambda x: x.abs().mean())
dist['share_abs_dev_ge2'] = df.groupby('cause').dev.apply(lambda x: (x.abs() >= 2).mean())
print('errors analysed:', len(df), 'of', int((te & (d != 0)).sum()), 'disagreement test logs')
print(dist.round(3).to_string())
dist.to_csv(f'{OUT_DIR}/error_taxonomy.csv')
df.to_csv(f'{OUT_DIR}/errors_all.csv', index=False)
df.sample(100, random_state=7).to_csv(f'{OUT_DIR}/errors_sample.csv', index=False)
json.dump({'n_errors': int(len(df)), 'n_dis_test': int((te & (d != 0)).sum()),
           'share_abs_dev_ge2': float((df.dev.abs() >= 2).mean())}, open(f'{OUT_DIR}/error_summary.json', 'w'))
