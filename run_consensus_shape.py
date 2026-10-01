"""Development analysis: how the next climber's deviation depends on the earlier logs of the route.
E[d | n earlier logs, their sum S] against the linear partial pooling prediction S / (n + lambda_route).
Computed on Protocol A training rows only, so it informs design without touching test data."""
import json
import numpy as np, pandas as pd
from common import *
g = load('indoor'); d = g.dev.values.astype(float)
S, n = prior_sum_count(g.route_id.values, g.t.values, d)
tr = split_protocol_a(g) == 'train'
lam = json.load(open(f'{OUT_DIR}/pace_lambdas_indoor.json'))['route']
df = pd.DataFrame({'n': n[tr], 'S': S[tr], 'd': d[tr]})
df = df[(df.n >= 1) & (df.n <= 5)]
t = df.groupby(['n', 'S']).d.agg(['mean', 'size']).reset_index()
t = t[t['size'] >= 100].copy()
t['linear'] = t.S / (t.n + lam)
t['unanimous'] = (t.S.abs() >= t.n)
print(t.round(3).to_string(index=False))
t.to_csv(f'{OUT_DIR}/consensus_shape.csv', index=False)
