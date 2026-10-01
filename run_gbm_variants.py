"""Flexible learner variants used to check that conclusions do not depend on PACE's linear form.
  B3-GBM[-climber]    B3 without any feature that describes the climber (their deviations, rates, level, grade gap)
  PACE-GBM            gradient boosting that starts from the PACE prediction (init_score) and learns a correction
                      from the extended features plus PACE's effects and evidence counts
  PACE-GBM[-climber]  the same without the climber: starts from model C and sees no climber information
All inputs are prequential, so PACE outputs are valid features for training rows as well.
Usage: python run_gbm_variants.py indoor"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from common import *
from andric import indoor_features, outdoor_features
from features import extended_features

setting = sys.argv[1]
SEEDS = [0, 1, 2, 3, 4]
g = load(setting); d = g.dev.values.astype(float)
EF = extended_features(g, indoor_features(g) if setting == 'indoor' else None)
PERSONAL = [c for c in EF.columns if c.startswith('climber') or c in ('rel_grade', 'andric_user_dev_sign')]
splits = {'A': split_protocol_a(g), 'B': split_protocol_b(g, setting)}
PP = {'A': np.load(f'{OUT_DIR}/pace_preds_{setting}.npz'), 'B': np.load(f'{OUT_DIR}/pace_preds_{setting}_B.npz')}
BASE = dict(objective='regression', learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            min_child_samples=50, reg_lambda=1.0, n_jobs=2, verbose=-1)
preds, log = {}, []


def design(proto, variant):
    P = PP[proto]
    if variant == 'B3-GBM[-climber]':
        return EF.drop(columns=PERSONAL), None
    if variant == 'PACE-GBM':
        F = EF.copy()
        for k in ('climber', 'route', 'env'):
            F[f'pace_{k}'] = P[f'PACE|val|{k}']; F[f'pace_n_{k}'] = P[f'PACE|cnt|{k}']
        F['pace_fixed'] = P['PACE|fixed']
        return F, P['PACE|pred']
    if variant == 'PACE-GBM[-climber]':
        F = EF.drop(columns=PERSONAL).copy()
        F['pace_c_pred'] = P['PACE[route+env]|pred']
        for k in ('route', 'env'):
            F[f'pace_n_{k}'] = P[f'PACE|cnt|{k}']
        return F, P['PACE[route+env]|pred']


def fit_predict(F, init, fit_idx, pred_idx, seed, leaves, rounds):
    m = lgb.LGBMRegressor(num_leaves=leaves, n_estimators=rounds, random_state=seed, **BASE)
    m.fit(F.iloc[fit_idx], d[fit_idx], init_score=None if init is None else init[fit_idx])
    out = m.predict(F.iloc[pred_idx])
    return out + (0 if init is None else init[pred_idx])


for proto, sp in splits.items():
    tr = np.where(sp == 'train')[0]; dv = np.where(sp == 'dev')[0]; te = np.where(sp == 'test')[0]
    fit_all = np.sort(np.concatenate([tr, dv]))
    if proto == 'A':
        schedule = [(fit_all, te)]
    else:
        month = g.date.dt.to_period('M').values
        schedule = [(np.arange(0, te[month[te] == m].min()), te[month[te] == m]) for m in pd.unique(month[te])]
    for variant in ('B3-GBM[-climber]', 'PACE-GBM', 'PACE-GBM[-climber]'):
        t0 = time.time()
        F, init = design(proto, variant)
        best = None
        for leaves in (15, 31, 63):
            m = lgb.LGBMRegressor(num_leaves=leaves, n_estimators=3000, random_state=0, **BASE)
            m.fit(F.iloc[tr], d[tr], init_score=None if init is None else init[tr],
                  eval_set=[(F.iloc[dv], d[dv])], eval_init_score=None if init is None else [init[dv]],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            pdv = m.predict(F.iloc[dv], num_iteration=m.best_iteration_) + (0 if init is None else init[dv])
            sc = float(np.sqrt(np.mean((d[dv] - pdv) ** 2)))
            log.append(dict(protocol=proto, model=variant, leaves=leaves, best_iter=int(m.best_iteration_), devRMSE=sc))
            if best is None or sc < best[0]:
                best = (sc, leaves, max(int(m.best_iteration_), 1))
        rounds = max(1, int(best[2] * len(fit_all) / len(tr)))
        for s in SEEDS:
            key = f'{proto}|{variant}|{s}'
            arr = np.full(len(g), np.nan)
            arr[dv] = fit_predict(F, init, tr, dv, s, best[1], best[2])
            for fi, pi in schedule:
                arr[pi] = fit_predict(F, init, fi, pi, s, best[1], rounds)
            preds[key] = arr.astype(np.float32)
        print(proto, variant, 'dev', round(best[0], 5), 'leaves', best[1], 'iters', best[2], round(time.time() - t0), 's', flush=True)
np.savez_compressed(f'{OUT_DIR}/gbm_variant_preds_{setting}.npz', **preds)
json.dump(log, open(f'{OUT_DIR}/gbm_variant_tuning_{setting}.json', 'w'), indent=1)
print('saved')
