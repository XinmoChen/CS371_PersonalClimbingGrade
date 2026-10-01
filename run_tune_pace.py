"""Tune PACE penalties (lambda = sigma^2 / tau^2) by prequential squared error on development rows only.
Protocol A dev rows: last 10% of each climber's training logs. Protocol B dev rows: Jul to Aug 2019.
Test rows are never scored here. Each protocol selects its penalties on its own development rows only.
Usage: python run_tune_pace.py indoor|outdoor base|refine GRID_JSON|inter"""
import sys, itertools, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import *
from pace_model import Block, prequential

setting = sys.argv[1]; stage = sys.argv[2]
g = load(setting)
spA = split_protocol_a(g); spB = split_protocol_b(g, setting)
d = g.dev.values.astype(float)
X = np.column_stack([np.ones(len(g)), g.grade_id.values - 13.0])
C = {'climber': codes(g.user_id.values), 'route': codes(g.route_id.values)}
if setting == 'indoor':
    C['env'] = codes(g.route_setter_id.values)
    C['env_grade'] = codes(g.route_setter_id.values, g.grade_id.values)
else:
    C['env'] = codes(pd.to_numeric(g.rock_type, errors='coerce').fillna(-1).values)
    C['env_grade'] = codes(pd.to_numeric(g.rock_type, errors='coerce').fillna(-1).values, g.grade_id.values)
C['climber_grade'] = codes(g.user_id.values, g.grade_id.values)


import os
PERIOD = int(os.environ.get('PACE_PERIOD_DAYS', '1'))
INTER_CG = [100, 300, 1000]      # climber by grade penalty (first grid 300, 3000, 30000 put the optimum at its lower edge)
INTER_EG = [30, 100, 300]        # setter (rock type) by grade penalty (first grid 100, 300, 1000, same reason)
if stage == 'inter' and len(sys.argv) > 3:                     # optional extension: '[cg list]' '[eg list]'
    INTER_CG = json.loads(sys.argv[3]); INTER_EG = json.loads(sys.argv[4])


def run(cfg):
    blocks = [Block(k, C[k], v) for k, v in cfg.items()]
    t0 = time.time()
    res = prequential(d, X, blocks, g.day.values // PERIOD, g.t.values)
    p = res['pred']
    out = dict(cfg)
    for name, sp in (('A', spA), ('B', spB)):
        m = sp == 'dev'
        out[f'devRMSE_{name}'] = float(np.sqrt(np.mean((d[m] - p[m]) ** 2)))
        out[f'devRMSE_r_{name}'] = float(np.sqrt(np.mean((d[m] - np.clip(np.round(p[m]), -3, 3)) ** 2)))
    out['sec'] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)
    return out


if __name__ == '__main__':
    if stage == 'refine':
        grid = json.loads(sys.argv[3])
    elif stage == 'base' and setting == 'outdoor':
        grid = [dict(climber=a, route=b, env=c) for a, b, c in itertools.product([20, 40], [3, 6, 12], [30, 300])]
    elif stage == 'base':
        grid = [dict(climber=a, route=b, env=c) for a, b, c in itertools.product([10, 20, 40], [1, 2, 4], [30, 100, 300, 1000])]
    else:
        # interaction penalties, searched separately around each protocol's own base penalties
        grid = []
        for suf in ('', '_B'):
            best = json.load(open(f'{OUT_DIR}/pace_best_base_{setting}{suf}.json'))
            base = {k: float(best[k]) for k in ('climber', 'route', 'env')}
            grid += [dict(base, climber_grade=float(a), env_grade=float(b)) for a, b in itertools.product(INTER_CG, INTER_EG)]
    with Pool(2) as pool:
        rows = pool.map(run, grid, chunksize=1)
    df = pd.DataFrame(rows)
    import os
    prev = f'{OUT_DIR}/pace_tuning_all_{setting}.csv'
    allrows = (pd.concat([pd.read_csv(prev), df], ignore_index=True) if os.path.exists(prev) else df).drop_duplicates(subset=[c for c in ('climber','route','env','climber_grade','env_grade') if c in df])
    allrows.to_csv(prev, index=False)
    df.to_csv(f'{OUT_DIR}/pace_tuning_{stage}_{setting}.csv', index=False)
    if stage == 'refine':
        base_cols = [c for c in ('climber', 'route', 'env') if c in allrows]
        b = allrows[allrows.columns.intersection(base_cols + ['devRMSE_A', 'devRMSE_B', 'devRMSE_r_A', 'devRMSE_r_B'])]
        full = allrows.dropna(subset=['devRMSE_A'])
        if 'climber_grade' in full:
            full = full[full['climber_grade'].isna()]
        full = full.reset_index(drop=True)
        for proto, suf in (('A', ''), ('B', '_B')):
            best = full.loc[full[f'devRMSE_{proto}'].idxmin()].to_dict()
            json.dump(best, open(f'{OUT_DIR}/pace_best_base_{setting}{suf}.json', 'w'))
            print('overall best base, protocol', proto, best)
        raise SystemExit
    if stage == 'inter':
        # choose among every interaction setting searched so far at this protocol's base penalties
        prev_inter = pd.read_csv(f'{OUT_DIR}/pace_tuning_inter_all_{setting}.csv') if os.path.exists(f'{OUT_DIR}/pace_tuning_inter_all_{setting}.csv') else df.iloc[:0]
        inter_all = pd.concat([prev_inter, df], ignore_index=True).drop_duplicates(subset=['climber', 'route', 'env', 'climber_grade', 'env_grade'], keep='last')
        inter_all.to_csv(f'{OUT_DIR}/pace_tuning_inter_all_{setting}.csv', index=False)
        for proto, suf in (('A', ''), ('B', '_B')):
            base = json.load(open(f'{OUT_DIR}/pace_best_base_{setting}{suf}.json'))
            own = inter_all[(inter_all.climber == float(base['climber'])) & (inter_all.route == float(base['route'])) & (inter_all.env == float(base['env']))]
            b = own.loc[own[f'devRMSE_{proto}'].idxmin()].to_dict()
            json.dump(b, open(f'{OUT_DIR}/pace_best_inter_{setting}{suf}.json', 'w'))
            print('best interaction penalties, protocol', proto, b)
    else:
        for proto, suf in (('A', ''), ('B', '_B')):
            b = df.loc[df[f'devRMSE_{proto}'].idxmin()].to_dict()
            json.dump(b, open(f'{OUT_DIR}/pace_best_{stage}_{setting}{suf}.json', 'w'))
            print('best by', proto, b)
