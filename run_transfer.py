"""RQ4: does a climber's grading tendency transfer between gym and crag?
For the climbers present in both files, test logs in one setting are predicted by that setting's PACE with its
personal effect replaced by (a) the same setting's effect (standard PACE), (b) no personal effect,
(c) the other setting's effect, or (d) a pooled effect that combines the evidence of both settings.
The other setting's effect comes from its day-start snapshot on or before the prediction day, so it only uses
logs dated earlier. Protocol A test rows; outdoor logs before the indoor data begins are excluded."""
import json
import numpy as np, pandas as pd
from common import *
from metrics import rmse, climber_bootstrap

out = {}
data = {s: load(s) for s in ('indoor', 'outdoor')}
P = {s: np.load(f'{OUT_DIR}/pace_preds_{s}.npz') for s in data}
S = {s: np.load(f'{OUT_DIR}/pace_snapshots_{s}.npz') for s in data}
LAM = {s: json.load(open(f'{OUT_DIR}/pace_lambdas_{s}.json')) for s in data}
both = sorted(set(data['indoor'].user_id) & set(data['outdoor'].user_id))
print('climbers in both settings:', len(both))
# code maps: snapshot effect arrays are indexed by codes(user_id) of each setting
cmap = {s: dict(zip(data[s].user_id.values, codes(data[s].user_id.values))) for s in data}
cnt_all = {}
DAYS_BY_CODE = {}
for s in data:
    g = data[s]; c = codes(g.user_id.values)
    cnt_all[s] = (c, g.day.values)
    order = np.argsort(c, kind='stable'); cs = c[order]; ds = g.day.values[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    DAYS_BY_CODE[s] = dict(zip(cs[np.r_[0, bounds]], np.split(ds, bounds)))   # sorted days per climber (rows are in time order)


def other_effect(target, source, users, days):
    """Personal effect of `users` in setting `source`, from the latest source snapshot strictly usable at `days`,
    plus the number of source logs behind it (logs on earlier source days)."""
    snap = S[source]; per = snap['period_start_day']
    # snapshot j holds the fit on all logs before its period start; the latest start <= day is usable
    j = np.searchsorted(per, days, side='right') - 1
    eff = np.zeros(len(users)); n = np.zeros(len(users))
    days_by_code = DAYS_BY_CODE[source]
    E = snap['eff_climber']
    for k, (u, jj) in enumerate(zip(users, j)):
        if jj < 0 or u not in cmap[source]:
            continue
        code = cmap[source][u]
        eff[k] = E[jj][code]
        n[k] = np.searchsorted(days_by_code[code], per[jj], side='left')   # source logs on earlier days
    return eff, n


rows = []
for target, source in (('outdoor', 'indoor'), ('indoor', 'outdoor')):
    g = data[target]; p = P[target]
    te = (split_protocol_a(g) == 'test') & g.user_id.isin(both).values
    if target == 'outdoor':
        te &= (g.date >= data['indoor'].date.min()).values
    idx = np.where(te)[0]
    u_own = p['PACE|val|climber'][idx]; n_own = p['PACE|cnt|climber'][idx]
    base = p['PACE|pred'][idx] - u_own
    u_src, n_src = other_effect(target, source, g.user_id.values[idx], g.day.values[idx])
    lam_t = LAM[target]['climber']; lam_s = LAM[source]['climber']
    pooled = (u_own * (n_own + lam_t) + u_src * (n_src + lam_s)) / (n_own + n_src + lam_t)
    y = g.dev.values[idx].astype(float); users = g.user_id.values[idx]
    preds = {'own': base + u_own, 'none': base, 'other': base + u_src, 'pooled': base + pooled}
    bs = climber_bootstrap(y, preds, users, B=1000, ref='none')
    dis = y != 0
    for k, pr in preds.items():
        rows.append(dict(target=target, effect=k, n_logs=int(len(idx)), n_climbers=int(len(set(users))),
                         RMSE=rmse(y, pr), RMSE_dis=rmse(y[dis], pr[dis]), lo=bs[k]['lo'], hi=bs[k]['hi'],
                         diff_vs_none=bs[k].get('diff', 0.0), diff_lo=bs[k].get('diff_lo', 0.0), diff_hi=bs[k].get('diff_hi', 0.0),
                         median_source_logs=float(np.median(n_src))))
    # how strongly does the other setting's effect predict this setting's residual? Slope of the residual after
    # community and context (y minus PACE without its personal effect) on the other setting's effect, fit on the
    # training and development rows of the same climbers, with a climber cluster bootstrap. The slope is then used
    # to scale the other setting's effect on the test rows ('other, scaled').
    sp = split_protocol_a(g)
    tr = np.isin(sp, ['train', 'dev']) & g.user_id.isin(both).values & (g.date >= data['indoor'].date.min()).values
    itr = np.where(tr)[0]
    x_tr, n_tr = other_effect(target, source, g.user_id.values[itr], g.day.values[itr])
    keep = n_tr > 0; itr, x_tr = itr[keep], x_tr[keep]
    r_tr = g.dev.values[itr].astype(float) - (p['PACE|pred'][itr] - p['PACE|val|climber'][itr])
    slope = lambda xx, rr: float(np.sum(xx * rr) / np.sum(xx * xx))
    b_tr = slope(x_tr, r_tr)
    uc, inv = np.unique(g.user_id.values[itr], return_inverse=True)
    sxy = np.bincount(inv, x_tr * r_tr); sxx = np.bincount(inv, x_tr * x_tr)
    Wb = np.random.default_rng(0).multinomial(len(uc), np.full(len(uc), 1.0 / len(uc)), size=1000).astype(float)
    bb = (Wb @ sxy) / (Wb @ sxx)
    r_te = y - base; ok = n_src > 0
    out[f'slope_{target}_on_{source}'] = dict(train=b_tr, train_lo=float(np.percentile(bb, 2.5)), train_hi=float(np.percentile(bb, 97.5)),
                                              train_climbers=int(len(uc)), test=slope(u_src[ok], r_te[ok]))
    sc = {'none': base, 'other, scaled': base + b_tr * u_src}
    bs2 = climber_bootstrap(y, sc, users, B=1000, ref='none')
    rows.append(dict(target=target, effect='other, scaled', n_logs=int(len(idx)), n_climbers=int(len(set(users))),
                     RMSE=rmse(y, sc['other, scaled']), RMSE_dis=rmse(y[dis], sc['other, scaled'][dis]),
                     lo=bs2['other, scaled']['lo'], hi=bs2['other, scaled']['hi'], diff_vs_none=bs2['other, scaled']['diff'],
                     diff_lo=bs2['other, scaled']['diff_lo'], diff_hi=bs2['other, scaled']['diff_hi'], median_source_logs=float(np.median(n_src))))
    # correlation of the latest personal effects across settings (joint PACE estimates, not raw means)
    last = pd.DataFrame({'u': users, 'own': u_own, 'src': u_src, 'n_src': n_src}).groupby('u').last()
    last = last[last.n_src > 0]
    out[f'corr_{target}_vs_{source}'] = dict(r=float(last.own.corr(last.src)), n=int(len(last)))
df = pd.DataFrame(rows)
print(df.round(4).to_string()); print(out)
df.to_csv(f'{OUT_DIR}/transfer.csv', index=False)
json.dump(out, open(f'{OUT_DIR}/transfer_corr.json', 'w'), indent=1)
