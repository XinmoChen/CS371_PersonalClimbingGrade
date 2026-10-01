"""Scores every model on the test rows of each protocol and writes the result tables used in the paper.
Outputs (results/):  main_<setting>_<protocol>.csv       headline metrics, mean and sd over seeds
                     boot_<setting>_<protocol>.json      climber cluster bootstrap CIs and paired differences
                     strata_<setting>_<protocol>.csv     RMSE by route history and climber history strata
                     helped_<setting>_<protocol>.csv     share of climbers helped or hurt by each personal term
                     gate_sweep_<setting>_<protocol>.csv PACE-G threshold sweep (dev and test)
                     shapley_<setting>_<protocol>.csv    Shapley attribution of MSE reduction to P, C, E
                     ordinal_<setting>_<protocol>.csv    class probability quality of the ordinal head
Usage: python run_evaluate.py indoor|outdoor"""
import sys, json, itertools
import numpy as np, pandas as pd
from math import factorial
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import log_loss
from common import *
from metrics import *
from pace_model import gated

setting = sys.argv[1]
g = load(setting); g = add_history_counts(g)
d = g.dev.values.astype(float); users = g.user_id.values
PS = {'A': dict(np.load(f'{OUT_DIR}/pace_preds_{setting}.npz')), 'B': dict(np.load(f'{OUT_DIR}/pace_preds_{setting}_B.npz'))}
LAMS = {'A': json.load(open(f'{OUT_DIR}/pace_lambdas_{setting}.json')), 'B': json.load(open(f'{OUT_DIR}/pace_lambdas_{setting}_B.json'))}
import os
Bz = dict(np.load(f'{OUT_DIR}/baseline_preds_{setting}.npz'))
if os.path.exists(f'{OUT_DIR}/gbm_variant_preds_{setting}.npz'):
    Bz.update(dict(np.load(f'{OUT_DIR}/gbm_variant_preds_{setting}.npz')))
ABL = dict(np.load(f'{OUT_DIR}/ablation_preds_{setting}.npz')) if os.path.exists(f'{OUT_DIR}/ablation_preds_{setting}.npz') else {}
splits = {'A': split_protocol_a(g), 'B': split_protocol_b(g, setting)}

Z = [0.0, 0.5, 1.0, 1.5, 1.96, 2.5, 3.0, 4.0, np.inf]


def model_preds(proto):
    """name -> list of prediction arrays (one per seed)."""
    out = {}
    for k, v in Bz.items():
        pr, name, seed = k.split('|')
        if pr == proto:
            out.setdefault(name, []).append(v.astype(np.float64))
    for k, v in PS[proto].items():
        name, kind = k.split('|')[0], k.split('|')[1]
        if kind == 'pred':
            out[name] = [v]
    for k, v in ABL.items():
        name = k.split('|')[0]
        if name[:-1] in ('PACE-static', 'PACE-sep', 'PACE-day'):      # per protocol variants
            if name[-1] != proto:
                continue
            name = name[:-1]
        out[name] = [v]
    return out


summary = {}
for proto, sp in splits.items():
    dev = sp == 'dev'; te = sp == 'test'
    P = PS[proto]; LAM = LAMS[proto]
    res_full = {'pred': P['PACE|pred'], 'block_value': {'climber': P['PACE|val|climber']},
                'block_count': {'climber': P['PACE|cnt|climber']}, 'sigma2': P['PACE|sigma2']}
    preds = model_preds(proto)
    # gate threshold chosen on dev rows by RMSE (ties broken toward larger z, i.e. fewer personal adjustments)
    sweep = []
    for z in Z:
        p, keep = gated(res_full, 'climber', LAM['climber'], z)
        row = dict(z=z)
        for nm, m in (('dev', dev), ('test', te)):
            row[f'RMSE_{nm}'] = rmse(d[m], p[m]); row[f'RMSE_r_{nm}'] = rmse(d[m], rounded(p[m]))
            row[f'share_personalized_{nm}'] = float(keep[m].mean())
            hh = helped_hurt(d[m], p[m], P['PACE|pred'][m] - P['PACE|val|climber'][m], users[m], tol=1e-3)
            row[f'hurt_{nm}'] = hh['hurt']; row[f'helped_{nm}'] = hh['helped']
        sweep.append(row)
    sweep = pd.DataFrame(sweep); sweep.to_csv(f'{OUT_DIR}/gate_sweep_{setting}_{proto}.csv', index=False)
    zbest = float(sweep.loc[sweep.RMSE_dev.round(6).idxmin(), 'z']) if True else 1.96
    # PACE-G uses the pre-registered 95% interval (z = 1.96); PACE-G* uses the dev-tuned z
    preds['PACE-noU'] = [P['PACE|pred'] - P['PACE|val|climber']]      # PACE with its personal effect removed
    preds['PACE-G'] = [gated(res_full, 'climber', LAM['climber'], 1.96)[0]]
    preds['PACE-G*'] = [gated(res_full, 'climber', LAM['climber'], zbest)[0]]
    summary[f'{proto}_zbest'] = zbest

    # ------------------------------------------------------------ headline metrics
    rows = []
    for name, arrs in preds.items():
        ms = [core_metrics(d[te], a[te], users[te]) for a in arrs if not np.isnan(a[te]).any()]
        if not ms:
            continue
        df = pd.DataFrame(ms)
        r = {'model': name, 'seeds': len(ms)}
        for c in df.columns:
            r[c] = df[c].mean(); r[c + '_sd'] = df[c].std(ddof=1) if len(ms) > 1 else 0.0
        rows.append(r)
    main = pd.DataFrame(rows).sort_values('RMSE')
    main.to_csv(f'{OUT_DIR}/main_{setting}_{proto}.csv', index=False)
    print(f'\n=== {setting} protocol {proto} (test n={te.sum()})')
    print(main[['model', 'seeds', 'RMSE', 'RMSE_sd', 'RMSE_r', 'MAE_r', 'RMSE_dis', 'userRMSE_r', 'macroF1', 'AP_dis']].round(4).to_string(index=False))

    # ------------------------------------------------------------ bootstrap CIs (per seed, then averaged over seeds)
    plist = {k: [a[te] for a in v] for k, v in preds.items() if not any(np.isnan(a[te]).any() for a in v)}
    mean_pred = {k: np.mean(np.stack(v), 0) for k, v in plist.items()}      # used only for deterministic models below
    ref = 'B1-RF' if 'B1-RF' in plist else 'B1-LR'
    boot = per_seed(lambda v: climber_bootstrap(d[te], v, users[te], B=1000, ref=ref), plist)
    boot_pace = per_seed(lambda v: climber_bootstrap(d[te], v, users[te], B=1000, ref='PACE[route+env]'), plist)
    json.dump({'ref_' + ref: boot, 'ref_PACE[route+env]': boot_pace}, open(f'{OUT_DIR}/boot_{setting}_{proto}.json', 'w'), indent=1)

    # ------------------------------------------------------------ strata
    srows = []
    rs = route_stratum(g.n_route.values); cs = climber_stratum(g.n_climber.values)
    gb = pd.cut(g.grade_id.values, [0, 11, 15, 19, 40], labels=['5a to 5c+', '6a to 6b+', '6c to 7a+', '7b and up']).astype(str)
    STRATA_LABELS = {'route': ROUTE_LABELS, 'climber': CLIMBER_LABELS, 'grade': ['5a to 5c+', '6a to 6b+', '6c to 7a+', '7b and up']}
    for kind, lab in (('route', rs), ('climber', cs), ('grade', gb)):
        for s in STRATA_LABELS[kind]:
            m = te & (lab == s)
            if m.sum() < 30:
                continue
            mt = m[te]; dt = d[te]; md = mt & (dt != 0)
            for name, arrs in plist.items():
                srows.append(dict(kind=kind, stratum=s, model=name, n=int(m.sum()), dis_rate=float((d[m] != 0).mean()),
                                  RMSE=float(np.mean([rmse(dt[mt], a[mt]) for a in arrs])),
                                  RMSE_r=float(np.mean([rmse(dt[mt], rounded(a[mt])) for a in arrs])),
                                  RMSE_dis=float(np.mean([rmse(dt[md], a[md]) for a in arrs])) if md.sum() else np.nan))
    pd.DataFrame(srows).to_csv(f'{OUT_DIR}/strata_{setting}_{proto}.csv', index=False)

    # ------------------------------------------------------------ helped / hurt by the personal term
    hrows = []
    pairs = [('PACE', 'PACE-noU'), ('PACE-G', 'PACE-noU'), ('PACE-G*', 'PACE-noU'), ('PACE', 'PACE[route+env]'),
             ('PACE-X', 'PACE[route+env]'), ('B1-LR', 'B1-LR[route+setter]' if setting == 'indoor' else 'B1-LR[route]'),
             ('B3-GBM', 'B3-GBM[-climber]'), ('PACE-GBM', 'PACE-GBM[-climber]')]
    for new, old in pairs:
        if new not in plist or old not in plist:
            continue
        for disp, f in (('continuous', lambda x: x), ('displayed', rounded)):
            for tol in (0.0, 1e-3):
                cf = tol > 0 and disp == 'continuous'
                hh = per_seed(lambda v: helped_hurt(d[te], f(v[new]), f(v[old]), users[te], tol=tol, crossfit=cf), {new: plist[new], old: plist[old]})
                hrows.append(dict(new=new, reference=old, output=disp, tol=tol, **hh))
    pd.DataFrame(hrows).to_csv(f'{OUT_DIR}/helped_{setting}_{proto}.csv', index=False)

    # ------------------------------------------------------------ Shapley attribution over P, C, E
    def shapley(names, players):
        mse = lambda S: float(np.mean((d[te] - mean_pred[names[tuple(sorted(S, key=players.index))]]) ** 2))
        base0 = float(np.mean(d[te] ** 2))                   # official grade
        shap = {}
        for pl in players:
            others = [p for p in players if p != pl]; val = 0.0
            for r in range(len(others) + 1):
                for S in itertools.combinations(others, r):
                    w = factorial(len(S)) * factorial(len(players) - len(S) - 1) / factorial(len(players))
                    val += w * (mse(S) - mse(S + (pl,)))
            shap[pl] = val
        total = base0 - mse(tuple(players))
        rows = [dict(source=k, mse_reduction=v) for k, v in shap.items()] + [dict(source='fixed (grade scale)', mse_reduction=base0 - mse(()))]
        out = pd.DataFrame(rows); out['share_of_total'] = out.mse_reduction / total
        sub = [dict(source='subset:' + '+'.join(S), mse_reduction=base0 - mse(S)) for S in names]
        return pd.concat([out, pd.DataFrame(sub)], ignore_index=True)
    pn = {(): 'PACE[]', ('climber',): 'PACE[climber]', ('route',): 'PACE[route]', ('env',): 'PACE[env]',
          ('climber', 'route'): 'PACE[climber+route]', ('climber', 'env'): 'PACE[climber+env]',
          ('route', 'env'): 'PACE[route+env]', ('climber', 'route', 'env'): 'PACE'}
    shapley(pn, ['climber', 'route', 'env']).to_csv(f'{OUT_DIR}/shapley_{setting}_{proto}.csv', index=False)

    def shapley_boot(names, players, Wb, inv, cnt):
        """Shapley shares of the full gain on every bootstrap draw (same climber weights for every model)."""
        den = Wb @ cnt
        mse = {S: (Wb @ np.bincount(inv, (d[te] - mean_pred[nm]) ** 2, minlength=len(cnt))) / den for S, nm in names.items()}
        base0 = (Wb @ np.bincount(inv, d[te] ** 2, minlength=len(cnt))) / den
        total = base0 - mse[tuple(players)]
        key = lambda S: tuple(sorted(S, key=players.index))
        phi = {}
        for pl in players:
            others = [p for p in players if p != pl]; val = 0.0
            for r in range(len(others) + 1):
                for S in itertools.combinations(others, r):
                    w = factorial(len(S)) * factorial(len(players) - len(S) - 1) / factorial(len(players))
                    val = val + w * (mse[key(S)] - mse[key(S + (pl,))])
            phi[pl] = val / total
        return phi
    uc, inv = np.unique(users[te], return_inverse=True); cntu = np.bincount(inv).astype(float)
    Wb = np.random.default_rng(0).multinomial(len(uc), np.full(len(uc), 1.0 / len(uc)), size=1000).astype(float)
    sb = {'PACE': shapley_boot(pn, ['climber', 'route', 'env'], Wb, inv, cntu)}
    groups = ['climber', 'route', 'setter'] if setting == 'indoor' else ['climber', 'route']
    bn = {}
    for r in range(len(groups) + 1):
        for S in itertools.combinations(groups, r):
            bn[S] = 'B1-LR[' + '+'.join(S) + ']' if len(S) < len(groups) else 'B1-LR'
    if all(v in mean_pred for v in bn.values()):
        # the B1-LR subset with no group still contains the official grade (and outdoor context columns)
        shapley(bn, groups).to_csv(f'{OUT_DIR}/shapley_sse_{setting}_{proto}.csv', index=False)
        sb['SSE'] = shapley_boot(bn, groups, Wb, inv, cntu)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    sbo = {f'{fam}|{pl}': q(v) for fam, dd in sb.items() for pl, v in dd.items()}
    if 'SSE' in sb:
        diff = sb['PACE']['climber'] - sb['SSE']['climber']
        sbo['PACE minus SSE personal share'] = [float(np.mean(diff))] + q(diff)
    json.dump(sbo, open(f'{OUT_DIR}/shapley_boot_{setting}_{proto}.json', 'w'), indent=1)

    # ------------------------------------------------------------ ordinal head (softer / same / harder)
    cls = np.sign(d).astype(int) + 1                          # 0 softer, 1 same, 2 harder
    orows = []
    for name in ['PACE', 'PACE-G', 'PACE-X', 'B1-RF', 'B3-GBM']:
        if name not in preds:
            continue
        from scipy.stats import norm
        # day start residual variance of each test row's own fit (earlier logs only)
        s2 = P['PACE|sigma2'][te]
        s2 = np.where(np.isnan(s2), np.nanmedian(P['PACE|sigma2'][dev]), s2)
        per = {'ordinal': [], 'gaussian': []}
        for a in preds[name]:                                  # one fit per seed, metrics averaged over seeds
            om = OrderedModel(pd.Series(cls[dev]), pd.DataFrame({'s': a[dev]}), distr='logit').fit(method='bfgs', disp=False)
            pr = np.asarray(om.predict(pd.DataFrame({'s': a[te]})))
            # Gaussian link alternative: P(class) from N(pred, sigma^2) with cut points -0.5, 0.5
            z1 = norm.cdf((-0.5 - a[te]) / np.sqrt(s2)); z2 = norm.cdf((0.5 - a[te]) / np.sqrt(s2))
            pg = np.column_stack([z1, z2 - z1, 1 - z2]).clip(1e-6, 1)
            pg = pg / pg.sum(1, keepdims=True)
            for link, prob in (('ordinal', pr), ('gaussian', pg)):
                per[link].append(dict(logloss=float(log_loss(cls[te], prob, labels=[0, 1, 2])),
                                      brier=float(np.mean(np.sum((prob - np.eye(3)[cls[te]]) ** 2, 1))),
                                      ece=expected_calibration_error(prob, cls[te]),
                                      AP_hard=float(average_precision_score(cls[te] == 2, prob[:, 2])),
                                      AP_soft=float(average_precision_score(cls[te] == 0, prob[:, 0])),
                                      AP_dis=float(average_precision_score(cls[te] != 1, 1 - prob[:, 1]))))
            if name == 'PACE':
                np.savez_compressed(f'{OUT_DIR}/ordinal_probs_{setting}_{proto}.npz', prob=pr, cls=cls[te])
        for link in ('ordinal', 'gaussian'):
            orows.append(dict(model=name, link=link, **average_outputs(per[link])))
    prior = np.bincount(cls[dev], minlength=3) / dev.sum()
    pp = np.tile(prior, (te.sum(), 1))
    orows.append(dict(model='class prior', link='-', logloss=float(log_loss(cls[te], pp, labels=[0, 1, 2])),
                      brier=float(np.mean(np.sum((pp - np.eye(3)[cls[te]]) ** 2, 1))), ece=expected_calibration_error(pp, cls[te])))
    pd.DataFrame(orows).to_csv(f'{OUT_DIR}/ordinal_{setting}_{proto}.csv', index=False)

json.dump(summary, open(f'{OUT_DIR}/eval_summary_{setting}.json', 'w'), indent=1)
print(summary)
