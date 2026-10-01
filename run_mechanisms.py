"""Numbers that explain why the ablations behave as they do (Section 5.1), indoor test rows of both protocols.
(a) Separate averages versus the joint fit: mean absolute prediction, and squared error on logs that agree and disagree.
(b) Same day update: share of test logs that follow an earlier log on the same route on the same day, and RMSE with
    and without the update on those logs and on the rest.
(c) Prior variance of route effects relative to climber effects implied by the tuned penalties (lambda = sigma^2 / tau^2).
Output: results/mechanisms_indoor.json."""
import json
import numpy as np, pandas as pd
from common import *

g = load('indoor'); d = g.dev.values.astype(float)
df = pd.DataFrame({'r': g.route_id.values, 'day': g.day.values, 't': g.t.values})
same_day = (df.t > df.groupby(['r', 'day']).t.transform('min')).values   # an earlier log on this route today
rmse = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
ab = np.load(f'{OUT_DIR}/ablation_preds_indoor.npz')
out = {}
for proto, sp, suf in (('A', split_protocol_a(g), ''), ('B', split_protocol_b(g, 'indoor'), '_B')):
    te = sp == 'test'; y = d[te]; dis = y != 0
    P = np.load(f'{OUT_DIR}/pace_preds_indoor{suf}.npz'); lam = json.load(open(f'{OUT_DIR}/pace_lambdas_indoor{suf}.json'))
    p = P['PACE|pred'][te]; s = ab[f'PACE-sep{proto}|pred'][te]; nd = ab[f'PACE-day{proto}|pred'][te]; m = same_day[te]
    out[proto] = {
        'mean_abs_pred': {'PACE': float(np.abs(p).mean()), 'PACE-sep': float(np.abs(s).mean())},
        'mse_agree': {'PACE': float(np.mean((y[~dis] - p[~dis]) ** 2)), 'PACE-sep': float(np.mean((y[~dis] - s[~dis]) ** 2))},
        'mse_disagree': {'PACE': float(np.mean((y[dis] - p[dis]) ** 2)), 'PACE-sep': float(np.mean((y[dis] - s[dis]) ** 2))},
        'share_test_after_same_day_log': float(m.mean()),
        'rmse_same_day': {'PACE': rmse(y[m], p[m]), 'PACE-day': rmse(y[m], nd[m])},
        'rmse_other': {'PACE': rmse(y[~m], p[~m]), 'PACE-day': rmse(y[~m], nd[~m])},
        'route_to_climber_prior_variance': lam['climber'] / lam['route'],
    }
print(json.dumps(out, indent=1))
json.dump(out, open(f'{OUT_DIR}/mechanisms_indoor.json', 'w'), indent=1)
