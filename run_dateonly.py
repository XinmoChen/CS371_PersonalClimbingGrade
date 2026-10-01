"""Sensitivity to logs that carry a date but no time of day (stamped 00:00:00).
The main protocol follows the released SQL (date < t), so such a log counts as earlier than every timed log of
its day. Here PACE is rerun with the conservative rule that a date only log becomes visible only from the next
day, and the test RMSE and the personal gain are compared. Output: results/dateonly_indoor.json."""
import json
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import *
from metrics import rmse
from pace_model import Block, prequential

g = load('indoor'); d = g.dev.values.astype(float)
X = np.column_stack([np.ones(len(g)), g.grade_id.values - 13.0])
C = {'climber': codes(g.user_id.values), 'route': codes(g.route_id.values), 'env': codes(g.route_setter_id.values)}
dateonly = (g.t.values % 86400 == 0)


def run(proto):
    suf = '' if proto == 'A' else '_B'
    LAM = json.load(open(f'{OUT_DIR}/pace_lambdas_indoor{suf}.json'))
    res = prequential(d, X, [Block(k, C[k], LAM[k]) for k in ('climber', 'route', 'env')], g.day.values, g.t.values, hide_dateonly=True)
    return proto, res['pred'], res['block_value']['climber']


if __name__ == '__main__':
    with Pool(2) as pool:
        out = pool.map(run, ['A', 'B'])
    # logs whose same day history contains a date only log of the same climber, route or setter
    exposed = np.zeros(len(g), bool)
    for k in ('climber', 'route', 'env'):
        s, n = prior_sum_count(np.stack([C[k], g.day.values], 1) @ np.array([1, 10 ** 7]), g.t.values, dateonly.astype(float))
        exposed |= s > 0
    res = {'share_dateonly_logs': float(dateonly.mean())}
    for proto, p, u in out:
        sp = split_protocol_a(g) if proto == 'A' else split_protocol_b(g, 'indoor')
        te = sp == 'test'
        P = np.load(f'{OUT_DIR}/pace_preds_indoor' + ('' if proto == 'A' else '_B') + '.npz')
        base, base_u = P['PACE|pred'], P['PACE|val|climber']
        res[proto] = dict(test_logs=int(te.sum()), exposed_test_logs=int((te & exposed).sum()),
                          RMSE_main=rmse(d[te], base[te]), RMSE_conservative=rmse(d[te], p[te]),
                          personal_gain_main=rmse(d[te], base[te] - base_u[te]) - rmse(d[te], base[te]),
                          personal_gain_conservative=rmse(d[te], p[te] - u[te]) - rmse(d[te], p[te]))
    print(json.dumps(res, indent=1))
    json.dump(res, open(f'{OUT_DIR}/dateonly_indoor.json', 'w'), indent=1)
