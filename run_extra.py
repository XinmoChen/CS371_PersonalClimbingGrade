"""Follow up analyses requested by the second audit round (indoor). Output: results/extra_indoor.json.
(a) Who is hurt by the uniform personal term: helped and hurt shares split by whether the climber has at least one
    test log that disagrees with the official grade, and the share of hurt climbers whose displayed error is unchanged.
(b) Share of the personal gain that the gate keeps (z = 1.96), with a climber cluster bootstrap interval.
(c) Route and setter nesting: number of routes with more than one setter."""
import json
import numpy as np, pandas as pd
from common import *
from metrics import rounded
from pace_model import gated

g = load('indoor'); d = g.dev.values.astype(float)
out = {'routes': int(g.route_id.nunique()), 'routes_with_more_than_one_setter': int((g.groupby('route_id').route_setter_id.nunique() > 1).sum())}
for proto, sp, suf in (('A', split_protocol_a(g), ''), ('B', split_protocol_b(g, 'indoor'), '_B')):
    te = sp == 'test'
    P = np.load(f'{OUT_DIR}/pace_preds_indoor{suf}.npz'); LAM = json.load(open(f'{OUT_DIR}/pace_lambdas_indoor{suf}.json'))
    p = P['PACE|pred'][te]; q = (P['PACE|pred'] - P['PACE|val|climber'])[te]; y = d[te]; u = g.user_id.values[te]
    df = pd.DataFrame({'u': u, 'dl': (y - p) ** 2 - (y - q) ** 2, 'dr': (y - rounded(p)) ** 2 - (y - rounded(q)) ** 2, 'dis': y != 0})
    per = df.groupby('u').agg(dl=('dl', 'mean'), dr=('dr', 'mean'), anydis=('dis', 'any'))
    hurt = per.dl > 1e-3; helped = per.dl < -1e-3
    res = {'climbers': int(len(per)), 'share_with_disagreement': float(per.anydis.mean())}
    for name, m in (('with_disagreement', per.anydis), ('all_agree', ~per.anydis)):
        res[name] = dict(n=int(m.sum()), helped=float(helped[m].mean()), hurt=float(hurt[m].mean()))
    res['hurt_with_unchanged_displayed_error'] = float((per.dr[hurt].abs() < 1e-12).mean())
    res['mean_mse_change_hurt'] = float(per.dl[hurt].mean()); res['mean_mse_change_helped'] = float(per.dl[helped].mean())
    # (b) gate retention with a climber bootstrap
    full = {'pred': P['PACE|pred'], 'block_value': {'climber': P['PACE|val|climber']}, 'block_count': {'climber': P['PACE|cnt|climber']}, 'sigma2': P['PACE|sigma2']}
    pg = gated(full, 'climber', LAM['climber'], 1.96)[0][te]
    uc, inv = np.unique(u, return_inverse=True); n = len(uc); cnt = np.bincount(inv).astype(float)
    sse = {k: np.bincount(inv, (y - v) ** 2, minlength=n) for k, v in (('pace', p), ('nou', q), ('gate', pg))}
    W = np.random.default_rng(0).multinomial(n, np.full(n, 1.0 / n), size=1000).astype(float)
    r = {k: np.sqrt((W @ s) / (W @ cnt)) for k, s in sse.items()}
    keep = (r['nou'] - r['gate']) / (r['nou'] - r['pace'])
    point = lambda k: np.sqrt(sse[k].sum() / cnt.sum())
    res['gate_keeps'] = dict(point=float((point('nou') - point('gate')) / (point('nou') - point('pace'))),
                             lo=float(np.percentile(keep, 2.5)), hi=float(np.percentile(keep, 97.5)))
    out[proto] = res
print(json.dumps(out, indent=1))
json.dump(out, open(f'{OUT_DIR}/extra_indoor.json', 'w'), indent=1)
