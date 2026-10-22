"""Refits of PACE for the design checks of the approach section (indoor, both protocols).

Two questions a reviewer could ask about simpler designs, each answered by refitting PACE prequentially:
  shrink  Instead of gating the personal effect, why not shrink it harder? PACE refit with larger climber penalties
          lambda_u (route and setter penalties unchanged), no gate.
  k4      Do the attribution shares depend on tuning the penalties? Every subset of {climber, route, env} refit with
          the fixed constant of the published model (lambda = 4 for every block), for a Shapley attribution.
Same data, ordering, prequential procedure and code path as run_pace.py.
Output: results/design_refits_indoor[_B].npz
Usage: python run_design_refits.py A|B"""
import sys, json, itertools, time
import numpy as np
from multiprocessing import Pool
from common import *
from pace_model import Block, prequential

proto = sys.argv[1]
SUF = '' if proto == 'A' else '_' + proto
g = load('indoor')
d = g.dev.values.astype(float)
X = np.column_stack([np.ones(len(g)), g.grade_id.values - 13.0])
C = {'climber': codes(g.user_id.values), 'route': codes(g.route_id.values), 'env': codes(g.route_setter_id.values)}
LAM = json.load(open(f'{OUT_DIR}/pace_lambdas_indoor{SUF}.json'))
period = g.day.values

CONFIGS = {}
for lu in (100, 200, 400, 1000, 3000):
    CONFIGS[f'shrink{lu}'] = {'climber': lu, 'route': LAM['route'], 'env': LAM['env']}
for r in range(0, 4):
    for sub in itertools.combinations(['climber', 'route', 'env'], r):
        CONFIGS['k4[' + '+'.join(sub) + ']'] = {k: 4.0 for k in sub}


def run(name):
    t0 = time.time()
    blocks = [Block(k, C[k], lam) for k, lam in CONFIGS[name].items()]
    res = prequential(d, X, blocks, period, g.t.values)
    print(name, 'done in', round(time.time() - t0, 1), 's', flush=True)
    out = {f'{name}|pred': res['pred']}
    if 'climber' in CONFIGS[name]:
        out[f'{name}|val|climber'] = res['block_value']['climber']
        out[f'{name}|cnt|climber'] = res['block_count']['climber']
        out[f'{name}|sigma2'] = res['sigma2']
    return out


if __name__ == '__main__':
    with Pool(2) as pool:
        results = pool.map(run, list(CONFIGS), chunksize=1)
    out = {}
    for r in results:
        out.update(r)
    np.savez_compressed(f'{OUT_DIR}/design_refits_indoor{SUF}.npz', **out)
    json.dump({k: v for k, v in CONFIGS.items()}, open(f'{OUT_DIR}/design_refits_configs_indoor{SUF}.json', 'w'), indent=1)
    print('saved')
