"""Verification gate V1: the prequential PACE predictions equal a from-scratch fit on strictly earlier logs.
Rows that are the first log of their day must match an exact refit on all earlier days to numerical precision.
For rows with same-day earlier logs, the one-step within-day update is compared with an exact refit on all
logs with a strictly earlier timestamp, and the approximation error is reported."""
import numpy as np, pandas as pd
from common import *
from pace_model import *
g = load('indoor')
g = g[g.date < '2018-12-01'].reset_index(drop=True)   # first three months keep the brute-force refits cheap
d = g.dev.values.astype(float); X = np.column_stack([np.ones(len(g)), g.grade_id.values - 13.0])
import json
LAM = json.load(open(f'{OUT_DIR}/pace_lambdas_indoor.json'))            # the tuned Protocol A penalties
mk = lambda: [Block('climber', codes(g.user_id.values), LAM['climber']), Block('route', codes(g.route_id.values), LAM['route']),
              Block('setter', codes(g.route_setter_id.values), LAM['env'])]
print('penalties', {k: LAM[k] for k in ('climber', 'route', 'env')}, 'rows', len(g))
res = prequential(d, X, mk(), g.day.values, g.t.values)
rng = np.random.default_rng(0)
first_of_day = np.r_[True, g.day.values[1:] != g.day.values[:-1]]
# (a) rows whose day has no strictly earlier same-day log for any of their keys: pick rows at first timestamp of day
first_ts = g.groupby('day').t.transform('min').values == g.t.values
cand_a = np.where(first_ts & (g.index.values > 500))[0]
errs_a = []
for n in rng.choice(cand_a, 40, replace=False):
    s = np.searchsorted(g.day.values, g.day.values[n])     # rows before this day
    blocks = mk(); (beta, eff), _, _ = fit_prefix(d, X, blocks, s, tol=1e-10, max_iter=5000)
    p = X[n] @ beta + sum(e[b.code[n]] for e, b in zip(eff, blocks))
    errs_a.append(abs(p - res['pred'][n]))
print('(a) first-timestamp rows: max |exact - prequential| =', np.max(errs_a))
# (b) rows with same-day earlier logs: compare with an exact refit on rows with t < t_n
cand_b = np.where(~first_ts & (g.index.values > 500))[0]
errs_b = []
for n in rng.choice(cand_b, 40, replace=False):
    m = g.t.values < g.t.values[n]; s = int(m.sum())       # rows sorted by time, so this is a prefix
    assert m[:s].all()
    blocks = mk(); (beta, eff), _, _ = fit_prefix(d, X, blocks, s, tol=1e-10, max_iter=5000)
    p = X[n] @ beta + sum(e[b.code[n]] for e, b in zip(eff, blocks))
    errs_b.append(abs(p - res['pred'][n]))
print('(b) rows with same-day history: mean |exact - prequential| = %.2e, max = %.2e' % (np.mean(errs_b), np.max(errs_b)))
# (c) no row uses its own label: perturb the labels of 5 random rows (one at a time) and confirm that no prediction
# at or before the perturbed row's timestamp changes, while some later prediction does
for n in rng.choice(np.arange(1000, len(g)), 5, replace=False):
    d2 = d.copy(); d2[n] += 3
    res2 = prequential(d2, X, mk(), g.day.values, g.t.values)
    later = g.t.values > g.t.values[n]; earlier_or_same = ~later
    print('(c) row', int(n), 'own change', abs(res2['pred'][n] - res['pred'][n]),
          '| max change at or before its timestamp', np.abs(res2['pred'] - res['pred'])[earlier_or_same].max(),
          '| some later prediction changes', bool(np.abs(res2['pred'] - res['pred'])[later].max() > 0))
