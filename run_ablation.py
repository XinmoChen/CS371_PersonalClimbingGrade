"""Ablations of PACE design choices (indoor). Each isolates one decision of Section 3.
  PACE-sep     separate shrunken means of the residual after the grade term, with PACE's penalties
               (no joint estimation): tests whether joint fitting matters
  PACE-k4      joint fit with every penalty fixed at k = 4 (the SSE constant): tests penalty tuning
  PACE-day     no within day update (only logs from earlier days): tests the same day information
  PACE-static  one fit on the training rows, no refitting. Protocol A: training rows include later logs of
               other climbers (the static protocol of the published CF model). Protocol B: fit on the
               training period only, so routes set later have no route effect.
Variants that use tuned penalties (sep, day, static) run once per protocol with that protocol's penalties
and fixed term, saved with a trailing A or B, so each row of the ablation table changes one choice only.
Usage: python run_ablation.py indoor"""
import sys, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import *
from pace_model import Block, prequential, fit_prefix

setting = sys.argv[1] if len(sys.argv) > 1 else 'indoor'
g = load(setting)
d = g.dev.values.astype(float)
X = np.column_stack([np.ones(len(g)), g.grade_id.values - 13.0])
env_key = g.route_setter_id.values if setting == 'indoor' else pd.to_numeric(g.rock_type, errors='coerce').fillna(-1).values
C = {'climber': codes(g.user_id.values), 'route': codes(g.route_id.values), 'env': codes(env_key)}
SUFS = {'A': '', 'B': '_B'}
LAMS = {p: json.load(open(f'{OUT_DIR}/pace_lambdas_{setting}{s}.json')) for p, s in SUFS.items()}
PS = {p: np.load(f'{OUT_DIR}/pace_preds_{setting}{s}.npz') for p, s in SUFS.items()}


def run(name):
    t0 = time.time()
    if name == 'PACE-k4':
        blocks = [Block(k, C[k], 4.0) for k in ('climber', 'route', 'env')]
        pred = prequential(d, X, blocks, g.day.values, g.t.values)['pred']
    elif name.startswith('PACE-day'):
        LAM = LAMS[name[-1]]
        blocks = [Block(k, C[k], LAM[k]) for k in ('climber', 'route', 'env')]
        # every log gets its own "time" equal to the day start, so no same day log is strictly earlier
        pred = prequential(d, X, blocks, g.day.values, g.day.values.astype(np.int64))['pred']
    elif name.startswith('PACE-sep'):
        LAM = LAMS[name[-1]]
        fixed = PS[name[-1]]['PACE|fixed']
        r = d - fixed
        pred = fixed.copy()
        for k in ('climber', 'route', 'env'):
            s, n = prior_sum_count(C[k], g.t.values, r)
            pred += s / (n + LAM[k])
    elif name.startswith('PACE-static'):
        proto = name[-1]
        sp = split_protocol_a(g) if proto == 'A' else split_protocol_b(g, setting)
        fit_rows = np.where(np.isin(sp, ['train', 'dev']))[0]
        idx = np.concatenate([fit_rows, np.setdiff1d(np.arange(len(g)), fit_rows)])
        L2 = json.load(open(f'{OUT_DIR}/pace_lambdas_{setting}' + ('' if proto == 'A' else '_B') + '.json'))
        blocks = [Block(k, C[k][idx], L2[k]) for k in ('climber', 'route', 'env')]
        (beta, eff), _, _ = fit_prefix(d[idx], X[idx], blocks, len(fit_rows), tol=1e-8, max_iter=2000)
        # codes are global, so effects index directly in the original row order
        pred = X @ beta + eff[0][C['climber']] + eff[1][C['route']] + eff[2][C['env']]
    print(name, round(time.time() - t0, 1), flush=True)
    return name, pred


if __name__ == '__main__':
    names = ['PACE-dayA', 'PACE-dayB', 'PACE-k4', 'PACE-sepA', 'PACE-sepB', 'PACE-staticA', 'PACE-staticB']
    with Pool(2) as pool:
        res = pool.map(run, names, chunksize=1)
    np.savez_compressed(f'{OUT_DIR}/ablation_preds_{setting}.npz', **{f'{n}|pred': p for n, p in res})
    print('saved')
