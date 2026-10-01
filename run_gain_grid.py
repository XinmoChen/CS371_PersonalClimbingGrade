"""Personal gain (RMSE of PACE without its personal effect minus RMSE of PACE) on test logs, cross classified by
route history and climber history at prediction time. Cells with fewer than 200 logs are left empty."""
import json
import numpy as np, pandas as pd
from common import *
from metrics import rmse, per_seed
g = add_history_counts(load('indoor')); d = g.dev.values.astype(float)
rows = []
for proto, sp, suf in (('A', split_protocol_a(g), ''), ('B', split_protocol_b(g, 'indoor'), '_B')):
    P = np.load(f'{OUT_DIR}/pace_preds_indoor{suf}.npz')
    p = P['PACE|pred']; q = p - P['PACE|val|climber']
    te = sp == 'test'
    rs = route_stratum(g.n_route.values); cs = climber_stratum(g.n_climber.values)
    for r in ROUTE_LABELS:
        for c in CLIMBER_LABELS:
            m = te & (rs == r) & (cs == c)
            rows.append(dict(protocol=proto, route=r, climber=c, n=int(m.sum()),
                             gain=(rmse(d[m], q[m]) - rmse(d[m], p[m])) if m.sum() >= 200 else np.nan))
df = pd.DataFrame(rows); df.to_csv(f'{OUT_DIR}/gain_grid_indoor.csv', index=False)
print(df.pivot_table(index=['protocol', 'route'], columns='climber', values='gain').round(4))


# Marginal route strata: personal gain of PACE (against PACE without its personal effect) and of the boosted models
# (against their refits without climber information), with climber cluster bootstrap intervals and the paired
# difference between new routes and routes with ten or more earlier logs. Seeded models: per seed, then averaged.
B = dict(np.load(f'{OUT_DIR}/baseline_preds_indoor.npz')); B.update(dict(np.load(f'{OUT_DIR}/gbm_variant_preds_indoor.npz')))
out = {}
srows = []
for proto, sp, suf in (('A', split_protocol_a(g), ''), ('B', split_protocol_b(g, 'indoor'), '_B')):
    te = sp == 'test'; P = np.load(f'{OUT_DIR}/pace_preds_indoor{suf}.npz')
    seeds = lambda name: [B[k].astype(float)[te] for k in sorted(B) if k.startswith(f'{proto}|{name}|')]
    fams = {'PACE': ([P['PACE|pred'][te]], [(P['PACE|pred'] - P['PACE|val|climber'])[te]]),
            'B3-GBM': (seeds('B3-GBM'), seeds('B3-GBM[-climber]')), 'PACE-GBM': (seeds('PACE-GBM'), seeds('PACE-GBM[-climber]'))}
    y = d[te]; users = g.user_id.values[te]; rs = route_stratum(g.n_route.values)[te]
    uc, inv = np.unique(users, return_inverse=True); nu = len(uc)
    W = np.random.default_rng(0).multinomial(nu, np.full(nu, 1.0 / nu), size=1000).astype(float)
    def gains(v):
        res = {}
        for s in ROUTE_LABELS:
            m = rs == s
            cnt = np.bincount(inv[m], minlength=nu).astype(float)
            e_new = np.bincount(inv[m], (y[m] - v['new'][m]) ** 2, minlength=nu); e_ref = np.bincount(inv[m], (y[m] - v['ref'][m]) ** 2, minlength=nu)
            den = W @ cnt
            res[s] = dict(point=float(np.sqrt(e_ref.sum() / cnt.sum()) - np.sqrt(e_new.sum() / cnt.sum())),
                          draws=np.sqrt((W @ e_ref) / den) - np.sqrt((W @ e_new) / den))
        diff = res[ROUTE_LABELS[0]]['draws'] - res[ROUTE_LABELS[-1]]['draws']
        o = {s: dict(gain=r['point'], lo=float(np.percentile(r['draws'], 2.5)), hi=float(np.percentile(r['draws'], 97.5))) for s, r in res.items()}
        o['new minus 10+'] = dict(gain=res[ROUTE_LABELS[0]]['point'] - res[ROUTE_LABELS[-1]]['point'],
                                  lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)))
        return o
    for fam, (new, ref) in fams.items():
        if not new or not ref:
            continue
        o = per_seed(gains, {'new': new, 'ref': ref})
        out[f'{proto}|{fam}'] = o
        print(proto, fam, {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in o.items()}, flush=True)
json.dump(out, open(f'{OUT_DIR}/gain_route_strata_indoor.json', 'w'), indent=1)
