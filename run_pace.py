"""Final PACE runs with the penalties chosen on development rows.

Runs (all prequential, every row predicted from strictly earlier logs):
  PACE      climber + route + env (env = setter indoors, rock type outdoors)
  PACE-X    PACE + climber x grade + env x grade interaction blocks
  subsets   every subset of {climber (P), route (C), env (E)} for the Shapley attribution of RQ1
Saves predictions, per-row effect values and evidence counts, and day-start snapshots for RQ2 and RQ4.
Usage: python run_pace.py indoor|outdoor [A|B] [period_days]"""
import sys, json, itertools, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import *
from pace_model import Block, prequential

setting = sys.argv[1]
proto = sys.argv[2] if len(sys.argv) > 2 else 'A'
period_days = int(sys.argv[3]) if len(sys.argv) > 3 else 1
SUF = '' if proto == 'A' else '_' + proto
g = load(setting)
d = g.dev.values.astype(float)
X = np.column_stack([np.ones(len(g)), g.grade_id.values - 13.0])
env_key = g.route_setter_id.values if setting == 'indoor' else pd.to_numeric(g.rock_type, errors='coerce').fillna(-1).values
C = {'climber': codes(g.user_id.values), 'route': codes(g.route_id.values), 'env': codes(env_key),
     'climber_grade': codes(g.user_id.values, g.grade_id.values), 'env_grade': codes(env_key, g.grade_id.values)}
period = g.day.values // period_days
best = json.load(open(f'{OUT_DIR}/pace_best_base_{setting}{SUF}.json'))
LAM = {k: float(best[k]) for k in ('climber', 'route', 'env')}
try:
    bi = json.load(open(f'{OUT_DIR}/pace_best_inter_{setting}{SUF}.json'))
    LAM.update({k: float(bi[k]) for k in ('climber_grade', 'env_grade')})
except FileNotFoundError:
    LAM.update({'climber_grade': 1e4, 'env_grade': 300.0})
json.dump(LAM, open(f'{OUT_DIR}/pace_lambdas_{setting}{SUF}.json', 'w'))

CONFIGS = {'PACE': ['climber', 'route', 'env'], 'PACE-X': ['climber', 'route', 'env', 'climber_grade', 'env_grade']}
for r in range(0, 3):
    for sub in itertools.combinations(['climber', 'route', 'env'], r):
        CONFIGS['PACE[' + '+'.join(sub) + ']'] = list(sub)


def run(name):
    t0 = time.time()
    blocks = [Block(k, C[k], LAM[k]) for k in CONFIGS[name]]
    res = prequential(d, X, blocks, period, g.t.values, snapshot_days=(name == 'PACE' and proto == 'A'))
    res['sec'] = time.time() - t0
    res['name'] = name
    print(name, 'done in', round(res['sec'], 1), 's', flush=True)
    return res


if __name__ == '__main__':
    with Pool(2) as pool:
        results = pool.map(run, list(CONFIGS), chunksize=1)
    out = {}
    for res in results:
        out[f"{res['name']}|pred"] = res['pred'].astype(np.float64)
        if res['name'] in ('PACE', 'PACE-X'):
            for k, v in res['block_value'].items():
                out[f"{res['name']}|val|{k}"] = v
            for k, v in res['block_count'].items():
                out[f"{res['name']}|cnt|{k}"] = v
            out[f"{res['name']}|sigma2"] = res['sigma2']; out[f"{res['name']}|fixed"] = res['fixed']
        if res['name'] == 'PACE' and proto == 'A':
            snaps = res['snapshots']; days = np.array(sorted(snaps))
            np.savez_compressed(f'{OUT_DIR}/pace_snapshots_{setting}.npz', periods=days, period_start_day=days * period_days,
                                beta=np.stack([snaps[k][0] for k in days]),
                                sigma2=np.array([snaps[k][2] for k in days]),
                                **{f'eff_{b}': np.stack([snaps[k][1][j] for k in days]) for j, b in enumerate(CONFIGS['PACE'])})
    np.savez_compressed(f'{OUT_DIR}/pace_preds_{setting}{SUF}.npz', **out)
    json.dump({r['name']: round(r['sec'], 1) for r in results}, open(f'{OUT_DIR}/pace_runtime_{setting}{SUF}.json', 'w'))
    print('saved')
