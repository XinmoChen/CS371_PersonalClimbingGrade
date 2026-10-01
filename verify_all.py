"""Verification gates. Each gate prints PASS or FAIL with the evidence. Run after the pipelines finish.
V1 prequential exactness and no self or future leakage (test_pace_exactness.py, run separately)
V2 determinism: a second PACE run returns identical predictions
V3 reproduction of the published baselines within 0.002 RMSE
V4 leakage guards: only allowed columns are loaded; extended features match brute force on sampled rows
V5 budget experiment with k = all equals PACE
V6 Shapley values plus the grade term sum to the total gain of PACE over the official grade
V7 metric sanity: official grade has macro F1 near 0.32 and identical rounded and continuous RMSE
V8 every bootstrap interval contains its point estimate
V9 the gate with z = 0 equals PACE and with z = infinity equals the no personal effect variant
"""
import json, sys
import numpy as np, pandas as pd
from common import *
from metrics import *
from pace_model import Block, prequential, gated
from features import extended_features

ok = True
def gate(name, cond, detail):
    global ok
    print(f'[{"PASS" if cond else "FAIL"}] {name}: {detail}'); ok &= bool(cond)

g = load('indoor'); d = g.dev.values.astype(float)
spA = split_protocol_a(g); te = spA == 'test'
P = np.load(f'{OUT_DIR}/pace_preds_indoor.npz'); LAM = json.load(open(f'{OUT_DIR}/pace_lambdas_indoor.json'))

# V2 determinism
if '--skip-rerun' not in sys.argv:
    X = np.column_stack([np.ones(len(g)), g.grade_id.values - 13.0])
    blocks = [Block('climber', codes(g.user_id.values), LAM['climber']), Block('route', codes(g.route_id.values), LAM['route']),
              Block('env', codes(g.route_setter_id.values), LAM['env'])]
    p2 = prequential(d, X, blocks, g.day.values, g.t.values)['pred']
    gate('V2 determinism', np.max(np.abs(p2 - P['PACE|pred'])) == 0.0, f'max abs diff {np.max(np.abs(p2 - P["PACE|pred"])):.2e}')

# V3 reproduction
main = pd.read_csv(f'{OUT_DIR}/main_indoor_A.csv').set_index('model')
for m, pub in (('B0-official', 0.418), ('B1-LR', 0.381), ('B1-RF', 0.378)):
    v = main.loc[m, 'RMSE_r']
    gate(f'V3 reproduction {m}', abs(v - pub) <= 0.002, f'ours {v:.4f} vs published {pub}')
v = main.loc['B0-official', 'userRMSE_r']; gate('V3 reproduction per climber RMSE', abs(v - 0.239) <= 0.002, f'{v:.4f} vs 0.239')

# V4 leakage guards
gate('V4 allowed columns only', set(g.columns) <= set(ALLOWED_INDOOR) | {'dev', 't', 'day', 'setting'}, sorted(g.columns))
EF = extended_features(g)
rng = np.random.default_rng(3); bad = 0
for n in rng.choice(len(g), 150, replace=False):
    row = g.iloc[n]; prior = g[(g.t < row.t)]
    r = prior[prior.route_id == row.route_id]; c = prior[prior.user_id == row.user_id]
    cg = c[c.grade_id == row.grade_id]; s = prior[prior.route_setter_id == row.route_setter_id]
    exp = dict(route_n=len(r), route_mean=r.dev.mean() if len(r) else 0.0, climber_n=len(c), climber_mean=c.dev.mean() if len(c) else 0.0,
               climber_grade_n=len(cg), setter_n=len(s), climber_level=c.grade_id.mean() if len(c) else np.nan)
    for k, v in exp.items():
        a = EF[k].iloc[n]
        if not ((np.isnan(v) and np.isnan(a)) or abs(a - v) < 1e-9):
            bad += 1
gate('V4 extended features brute force', bad == 0, f'{bad} mismatches on 150 sampled rows x 7 features')

# V5 budget
b = pd.read_csv(f'{OUT_DIR}/budget_indoor.csv')
v = b.loc[b.k.astype(str) == 'all', 'max_abs_diff_from_PACE'].iloc[0]
gate('V5 budget k=all equals PACE', v < 1e-5, f'max abs diff {v:.2e}')

# V6 Shapley efficiency
sh = pd.read_csv(f'{OUT_DIR}/shapley_indoor_A.csv')
tot_players = sh[sh.source.isin(['climber', 'route', 'env'])].mse_reduction.sum()
full = sh[sh.source == 'subset:climber+route+env'].mse_reduction.iloc[0]; fixed = sh[sh.source == 'fixed (grade scale)'].mse_reduction.iloc[0]
gate('V6 Shapley efficiency', abs(tot_players + fixed - full) < 1e-9, f'players {tot_players:.6f} + grade {fixed:.6f} = {tot_players + fixed:.6f} vs total {full:.6f}')

# V7 metric sanity
m0 = core_metrics(d[te], np.zeros(te.sum()), g.user_id.values[te])
gate('V7 official grade macro F1', abs(m0['macroF1'] - 0.32) < 0.01, f'{m0["macroF1"]:.4f}')
gate('V7 official grade RMSE = RMSE_r', abs(m0['RMSE'] - m0['RMSE_r']) < 1e-12, f'{m0["RMSE"]:.4f}')

# V8 bootstrap
bt = json.load(open(f'{OUT_DIR}/boot_indoor_A.json'))
viol = [(r, k) for r, dd in bt.items() for k, v in dd.items() if not (v['lo'] - 1e-12 <= v['RMSE'] <= v['hi'] + 1e-12)]
gate('V8 bootstrap intervals contain estimates', not viol, f'violations: {viol[:3]}')

# V9 gate endpoints
res = {'pred': P['PACE|pred'], 'block_value': {'climber': P['PACE|val|climber']}, 'block_count': {'climber': P['PACE|cnt|climber']}, 'sigma2': P['PACE|sigma2']}
p0, _ = gated(res, 'climber', LAM['climber'], 0.0); pinf, _ = gated(res, 'climber', LAM['climber'], np.inf)
gate('V9 gate z=0 equals PACE', np.max(np.abs(p0 - P['PACE|pred'])) < 1e-12, '')
gate('V9 gate z=inf removes the personal effect', np.max(np.abs(pinf - (P['PACE|pred'] - P['PACE|val|climber']))) < 1e-12, '')
print('ALL GATES PASS' if ok else 'SOME GATES FAILED')
