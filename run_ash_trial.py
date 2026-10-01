"""Adaptive shrinkage trial (Section 4.6, Unusual phenomena): replaces the Gaussian prior of the route effect with an
adaptive shrinkage mixture prior (Stephens 2017), fits its weights by empirical Bayes on Protocol A training rows,
then runs the prequential pass. Usage: python run_ash_trial.py route. The development RMSE it prints (0.38461 for
'route') is the number quoted in the paper, against 0.3683 for PACE, so the prior was dropped.
S2 is the fixed noise variance used in this exploratory trial."""
import numpy as np, json, time, sys
from common import *
from pace_model import Block, prequential
from eb_ash import fit_ash_weights
g = load('indoor'); d = g.dev.values.astype(float); X = np.column_stack([np.ones(len(g)), g.grade_id.values - 13.0])
sp = split_protocol_a(g); trn = np.where(sp == 'train')[0]
# static EB on training rows (reordered prefix)
gg = g.iloc[trn]
Xt = X[trn]; dt = d[trn]
LAM = json.load(open('results/pace_lambdas_indoor.json'))
S2 = 0.1166
cfg = sys.argv[1] if len(sys.argv) > 1 else 'all'
def blocks_for(codes_fn, which):
    out = []
    for k, key in (('climber', 'user_id'), ('route', 'route_id'), ('env', 'route_setter_id')):
        if k in which:
            out.append(Block(k, codes_fn(key), LAM[k], prior='ash', sigma2=S2, pi=np.full(8, 1/8)))
        else:
            out.append(Block(k, codes_fn(key), LAM[k]))
    return out
full_codes = {key: codes(g[key].values) for key in ('user_id', 'route_id', 'route_setter_id')}
which = {'all': ['climber', 'route', 'env'], 'route': ['route'], 'climber': ['climber'], 'cr': ['climber', 'route']}[cfg]
bt = blocks_for(lambda key: full_codes[key][trn], which)
t0 = time.time(); fit_ash_weights(dt, Xt, bt, len(trn), rounds=4, verbose=True); print('EB time', round(time.time()-t0))
bf = blocks_for(lambda key: full_codes[key], which)
for b, b2 in zip(bf, bt):
    b.pi = b2.pi
t0 = time.time(); res = prequential(d, X, bf, g.day.values, g.t.values); print('preq time', round(time.time()-t0))
p = res['pred']
for part in ('dev', 'test'):
    m = sp == part
    print(cfg, part, 'RMSE', round(float(np.sqrt(np.mean((d[m]-p[m])**2))), 5), 'RMSE_r', round(float(np.sqrt(np.mean((d[m]-np.clip(np.round(p[m]),-3,3))**2))), 5))
np.save(f'results/scratch_ash_{cfg}.npy', p)
json.dump({b.name: b.pi.tolist() for b in bf if b.prior == 'ash'}, open(f'results/scratch_ash_pi_{cfg}.json', 'w'))
