"""Design checks for the approach section (indoor, Protocols A and B). Output: results/design_checks_indoor.json,
results/gate_family_indoor_<protocol>.csv and results/paired_strata_indoor.csv.

Each check answers a question a reviewer could ask about a simpler design or about where the advanced model helps.
  1. Gate family. The evidence gate ranks logs by |u| / sd(u). Two simpler rankings use one ingredient each:
     a count gate (the climber's number of earlier logs) and a magnitude gate (|u| alone). A third alternative
     shrinks every personal effect harder (PACE refit with a larger lambda_u, run_design_refits.py) instead of
     gating. Rules are compared (a) at equal coverage, the share of test logs that receive a personal effect,
     which uses no labels, and (b) with each rule's threshold tuned on development rows, as the gate's z is.
     Reference for gains, helped and hurt: PACE with its personal effect removed, as in the paper.
  2. Where the advanced model differs from its baseline. Paired climber cluster bootstrap of RMSE differences
     (PACE minus the published forest B1-RF and linear regression B1-LR, PACE-GBM minus plain boosting B3) on all
     test logs, by route history, by climber history and by label. Seeded models are scored per seed and averaged.
  3. Evidence needed by the gate. Under the model, a climber with true effect u* and N earlier logs has
     u_hat ~ N(N u* / (N + lam), N sigma^2 / (N + lam)^2), and the gate keeps u_hat when
     |u_hat| > z sigma / sqrt(N + lam). This gives the logs needed before an effect of a given size is applied,
     and, with u* ~ N(0, tau^2) and tau^2 = sigma^2 / lam, the expected share of logs the gate personalizes by N,
     which is compared with the observed share.
  Exploratory: a track record gate keeps the personal effect when it lowered the error on the climber's own
     earlier logs (summed over those logs), which uses only labels of strictly earlier logs.
  Also: the mean size of each model's predicted deviation on agreeing and disagreeing test logs.
  4. Attribution with fixed penalties. Shapley shares of the test MSE reduction with every penalty fixed at the
     published constant k = 4 (run_design_refits.py), next to the tuned shares.
Usage: python run_design_checks.py [A|B ...]   (default: both protocols)"""
import json, sys
from math import factorial
import numpy as np, pandas as pd
from scipy.stats import norm
from common import *
from metrics import rmse, rounded, climber_bootstrap, helped_hurt, per_seed

g = load('indoor'); g = add_history_counts(g)
d = g.dev.values.astype(float); users = g.user_id.values
rs = route_stratum(g.n_route.values); cs = climber_stratum(g.n_climber.values)
B = 1000
COVERAGES = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
K_GRID = [0, 5, 10, 20, 50, 100, 200, 400, 800, 10 ** 9]
T_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, np.inf]
Z_GRID = [0.0, 0.5, 1.0, 1.5, 1.96, 2.5, 3.0, 4.0, np.inf]
out = {}
PROTOS = [a for a in sys.argv[1:] if a in ('A', 'B')] or ['A', 'B']
SUFS = {'A': '', 'B': '_B'}


def retention_boot(y, p_ref, p_full, p_rule, u, seed=0):
    """Share of the personal gain (RMSE of p_ref minus RMSE of p_full) kept by p_rule, with a climber bootstrap."""
    uc, inv = np.unique(u, return_inverse=True); n = len(uc); cnt = np.bincount(inv).astype(float)
    sse = {k: np.bincount(inv, (y - v) ** 2, minlength=n) for k, v in (('ref', p_ref), ('full', p_full), ('rule', p_rule))}
    W = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)
    r = {k: np.sqrt((W @ s) / (W @ cnt)) for k, s in sse.items()}
    keep = (r['ref'] - r['rule']) / (r['ref'] - r['full'])
    pt = lambda k: np.sqrt(sse[k].sum() / cnt.sum())
    return dict(point=float((pt('ref') - pt('rule')) / (pt('ref') - pt('full'))),
                lo=float(np.percentile(keep, 2.5)), hi=float(np.percentile(keep, 97.5)))


def share_diff_boot(y, p_a, p_b, p_ref, u, tol=1e-3, seed=0):
    """Paired climber bootstrap of the difference in helped and hurt shares between rules a and b (a minus b),
    each against the same reference, with the tolerance of the paper."""
    df = pd.DataFrame({'u': u, 'a': (y - p_a) ** 2 - (y - p_ref) ** 2, 'b': (y - p_b) ** 2 - (y - p_ref) ** 2})
    per = df.groupby('u')[['a', 'b']].mean()
    ind = {'hurt': (per.a > tol).astype(float).values - (per.b > tol).astype(float).values,
           'helped': (per.a < -tol).astype(float).values - (per.b < -tol).astype(float).values}
    n = len(per); W = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=B).astype(float) / n
    return {k: dict(diff=float(v.mean()), lo=float(np.percentile(W @ v, 2.5)), hi=float(np.percentile(W @ v, 97.5)))
            for k, v in ind.items()}


def rule_row(name, param, keep, p_rule, m, p_ref, p_full, tag):
    hh = helped_hurt(d[m], p_rule[m], p_ref[m], users[m], tol=1e-3)
    hd = helped_hurt(d[m], rounded(p_rule[m]), rounded(p_ref[m]), users[m], tol=1e-3)
    gain_full = rmse(d[m], p_ref[m]) - rmse(d[m], p_full[m])
    return dict(rule=name, param=param, rows=tag, coverage=float(keep[m].mean()), RMSE=rmse(d[m], p_rule[m]),
                retained=float((rmse(d[m], p_ref[m]) - rmse(d[m], p_rule[m])) / gain_full),
                helped=hh['helped'], hurt=hh['hurt'], helped_displayed=hd['helped'], hurt_displayed=hd['hurt'])


for proto, suf in ((p, SUFS[p]) for p in PROTOS):
    sp = split_protocol_a(g) if proto == 'A' else split_protocol_b(g, 'indoor')
    dev = sp == 'dev'; te = sp == 'test'
    P = np.load(f'{OUT_DIR}/pace_preds_indoor{suf}.npz'); LAM = json.load(open(f'{OUT_DIR}/pace_lambdas_indoor{suf}.json'))
    R = np.load(f'{OUT_DIR}/design_refits_indoor{suf}.npz')
    p_full = P['PACE|pred']; u = P['PACE|val|climber']; n = P['PACE|cnt|climber']
    s2 = np.nan_to_num(P['PACE|sigma2'], nan=np.nanmedian(P['PACE|sigma2']))
    lam = LAM['climber']; sd = np.sqrt(s2 / (n + lam))
    p_ref = p_full - u
    gate = lambda keep: p_ref + np.where(keep, u, 0.0)
    # exploratory: a climber's own track record, the summed gain of the personal effect on the climber's strictly
    # earlier logs (prequential predictions, labels of earlier logs only)
    gain_log = (d - p_ref) ** 2 - (d - p_full) ** 2
    track, _ = prior_sum_count(users, g.t.values, gain_log)
    score = {'evidence gate': np.abs(u) / sd, 'count gate': n.astype(float), 'magnitude gate': np.abs(u),
             'track record gate': track}
    res = {'lambda_u': lam}

    # ---------------------------------------------------------------- 1a. equal coverage (labels unused)
    rows = []
    z196 = np.abs(u) > 1.96 * sd
    cov_z = float(z196[te].mean())
    for name, s in score.items():
        for c in COVERAGES + [cov_z]:
            if c >= 1.0:
                keep = np.ones(len(u), bool)
            else:
                thr = np.quantile(s[te], 1 - c)          # threshold that personalizes a share c of test logs
                keep = s > thr
            rows.append(rule_row(name, float(c), keep, gate(keep), te, p_ref, p_full, 'test, equal coverage'))
    for key in sorted(k for k in R.files if k.startswith('shrink') and k.endswith('|pred')):
        lu = int(key.split('|')[0][6:])
        rows.append(rule_row('stronger shrinkage', lu, np.ones(len(u), bool), R[key], te, p_ref, p_full, 'test, lambda_u'))
    rows.append(rule_row('stronger shrinkage', lam, np.ones(len(u), bool), p_full, te, p_ref, p_full, 'test, lambda_u'))
    fam = pd.DataFrame(rows)
    fam.to_csv(f'{OUT_DIR}/gate_family_indoor_{proto}.csv', index=False)

    # ---------------------------------------------------------------- 1b. each rule tuned on development rows
    tuned = {}
    grids = {'evidence gate': ('z', Z_GRID, lambda v: np.abs(u) > v * sd),
             'count gate': ('K', K_GRID, lambda v: n >= v),
             'magnitude gate': ('t', T_GRID, lambda v: np.abs(u) > v),
             'track record gate': ('r', [-np.inf, 0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf], lambda v: track > v)}
    for name, (pname, grid, rule) in grids.items():
        best = None
        for v in grid:
            keep = rule(v); r_dev = round(rmse(d[dev], gate(keep)[dev]), 6); cov = float(keep[dev].mean())
            if best is None or r_dev < best[0] or (r_dev == best[0] and cov < best[2]):   # ties: fewer adjustments
                best = (r_dev, v, cov)
        keep = rule(best[1])
        row = rule_row(name, float(best[1]), keep, gate(keep), te, p_ref, p_full, 'test, tuned on dev')
        row.update(param_name=pname, dev_RMSE=best[0], retained_ci=retention_boot(d[te], p_ref[te], p_full[te], gate(keep)[te], users[te]))
        tuned[name] = row
    best = None
    for key in sorted(k for k in R.files if k.startswith('shrink') and k.endswith('|pred')) + ['base']:
        p = p_full if key == 'base' else R[key]; lu = lam if key == 'base' else int(key.split('|')[0][6:])
        r_dev = round(rmse(d[dev], p[dev]), 6)
        if best is None or r_dev < best[0] or (r_dev == best[0] and lu > best[1]):
            best = (r_dev, lu, p)
    row = rule_row('stronger shrinkage', best[1], np.ones(len(u), bool), best[2], te, p_ref, p_full, 'test, tuned on dev')
    row.update(param_name='lambda_u', dev_RMSE=best[0])
    tuned['stronger shrinkage'] = row
    res['tuned_on_dev'] = tuned

    # ---------------------------------------------------------------- 1c. paired comparison at the gate's coverage
    cmp = {'evidence gate': gate(z196)}
    for name in ('count gate', 'magnitude gate', 'track record gate'):
        thr = np.quantile(score[name][te], 1 - cov_z); cmp[name] = gate(score[name] > thr)
    # stronger shrinkage: the lambda_u whose test hurt share is closest to the gate's, for a like for like harm level
    hz = helped_hurt(d[te], cmp['evidence gate'][te], p_ref[te], users[te], tol=1e-3)['hurt']
    sh = [(abs(r.hurt - hz), r.param) for r in fam[(fam.rule == 'stronger shrinkage')].itertuples()]
    lu_match = min(sh)[1]
    cmp['stronger shrinkage'] = p_full if lu_match == lam else R[f'shrink{int(lu_match)}|pred']
    cmp['PACE'] = p_full; cmp['PACE without personal effect'] = p_ref
    bt = climber_bootstrap(d[te], {k: v[te] for k, v in cmp.items()}, users[te], B=B, ref='evidence gate')
    shares = {k: share_diff_boot(d[te], cmp['evidence gate'][te], cmp[k][te], p_ref[te], users[te])
              for k in ('count gate', 'magnitude gate', 'track record gate', 'stronger shrinkage')}
    res['equal_coverage'] = dict(coverage=cov_z, shrinkage_lambda_matched_on_hurt=float(lu_match), bootstrap=bt,
                                 share_diff_evidence_minus_rule=shares,
                                 retained={k: retention_boot(d[te], p_ref[te], p_full[te], v[te], users[te]) for k, v in cmp.items()
                                           if k in ('evidence gate', 'count gate', 'magnitude gate', 'track record gate', 'stronger shrinkage')})

    # ---------------------------------------------------------------- 3. evidence the gate needs
    sig2 = float(np.median(s2[te])); sig = np.sqrt(sig2); tau = sig / np.sqrt(lam); z = 1.96

    def p_pass(N, ustar):
        mu = N * ustar / (N + lam); s = np.sqrt(N * sig2) / (N + lam); thr = z * sig / np.sqrt(N + lam)
        return norm.sf((thr - mu) / s) + norm.cdf((-thr - mu) / s)

    def logs_needed(ustar, prob):
        for N in range(1, 200001):
            if p_pass(N, ustar) >= prob:
                return N
        return None
    need = {f'{us:.3f}': {'half': logs_needed(us, 0.5), 'eighty': logs_needed(us, 0.8)}
            for us in (tau, 2 * tau, 0.1, 0.2, 0.3, 0.5)}
    bins = [0, 10, 20, 50, 100, 200, 400, 10 ** 9]
    nb = pd.cut(n[te], bins, right=False)
    pred_share = 2 * norm.sf(z * sig * np.sqrt(n[te] + lam) / np.sqrt(n[te] ** 2 * tau ** 2 + n[te] * sig2 + 1e-12))
    by_n = pd.DataFrame({'bin': nb.astype(str), 'obs': z196[te], 'pred': pred_share}).groupby('bin', sort=False).agg(
        logs=('obs', 'size'), observed=('obs', 'mean'), predicted=('pred', 'mean'))
    res['gate_evidence'] = dict(sigma=sig, tau=tau, z=z, lambda_u=lam, logs_needed=need,
                                by_history=by_n.reset_index().to_dict(orient='records'),
                                overall_observed=float(z196[te].mean()), overall_predicted=float(pred_share.mean()),
                                median_climber_history_test=float(np.median(n[te])))
    # hypothetical scenarios: lower noise or larger tendencies, logs needed for a typical tendency (u* = tau)
    hyp = {}
    for label, sig_mult, tau_mult in (('as estimated', 1, 1), ('noise halved', 0.5, 1), ('tendencies doubled', 1, 2)):
        s2h = sig2 * sig_mult ** 2; th = tau * tau_mult; lam_h = s2h / th ** 2

        def pp(N, s2h=s2h, th=th, lam_h=lam_h):
            mu = N * th / (N + lam_h); s = np.sqrt(N * s2h) / (N + lam_h); thr = z * np.sqrt(s2h) / np.sqrt(N + lam_h)
            return norm.sf((thr - mu) / s) + norm.cdf((-thr - mu) / s)
        hyp[label] = next((N for N in range(1, 200001) if pp(N) >= 0.5), None)
    res['gate_evidence']['hypothetical'] = {'logs for a typical tendency (u* = tau) to pass with probability 0.5': hyp}

    # ---------------------------------------------------------------- 4. attribution with fixed penalties
    def shap(names):
        players = ['climber', 'route', 'env']
        mse = lambda S: float(np.mean((d[te] - names[tuple(p for p in players if p in S)][te]) ** 2))
        full = tuple(players); base0 = mse(())
        phi = {}
        for j in players:
            tot = 0.0
            others = [p for p in players if p != j]
            for r in range(0, 3):
                for S in __import__('itertools').combinations(others, r):
                    w = factorial(len(S)) * factorial(2 - len(S)) / factorial(3)
                    tot += w * ((base0 - mse(S + (j,))) - (base0 - mse(S)))
            phi[j] = tot
        total = base0 - mse(full)
        return {k: dict(value=v, share=v / (mse_official - mse(full))) for k, v in phi.items()} | {'total_vs_grade_term': total}
    mse_official = float(np.mean(d[te] ** 2))
    k4 = {tuple(k.split('|')[0][3:-1].split('+')) if k.split('|')[0] != 'k4[]' else (): R[k] for k in R.files if k.startswith('k4[') and k.endswith('|pred')}
    tuned_preds = {(): P['PACE[]|pred'], ('climber',): P['PACE[climber]|pred'], ('route',): P['PACE[route]|pred'], ('env',): P['PACE[env]|pred'],
                   ('climber', 'route'): P['PACE[climber+route]|pred'], ('climber', 'env'): P['PACE[climber+env]|pred'],
                   ('route', 'env'): P['PACE[route+env]|pred'], ('climber', 'route', 'env'): P['PACE|pred']}
    res['attribution'] = {'tuned': shap(tuned_preds), 'fixed_k4': shap(k4),
                          'RMSE_full_tuned': rmse(d[te], P['PACE|pred'][te]), 'RMSE_full_k4': rmse(d[te], k4[('climber', 'route', 'env')][te])}
    out[proto] = res
    print(proto, 'gate family and attribution done', flush=True)

# -------------------------------------------------------------------- 2. paired comparisons by group
Bz = dict(np.load(f'{OUT_DIR}/baseline_preds_indoor.npz')); Bz.update(dict(np.load(f'{OUT_DIR}/gbm_variant_preds_indoor.npz')))
prow = []
for proto, suf in ((p, SUFS[p]) for p in PROTOS):
    sp = split_protocol_a(g) if proto == 'A' else split_protocol_b(g, 'indoor'); te = sp == 'test'
    P = np.load(f'{OUT_DIR}/pace_preds_indoor{suf}.npz')
    seeded = lambda name: [Bz[f'{proto}|{name}|{s}'].astype(np.float64) for s in range(5)]
    models = {'PACE': [P['PACE|pred']], 'B1-RF': seeded('B1-RF'), 'B1-LR': [Bz[f'{proto}|B1-LR|0'].astype(np.float64)],
              'B3-GBM': seeded('B3-GBM'), 'PACE-GBM': seeded('PACE-GBM')}
    # size of the predicted deviation, mean |d_hat| on agreeing and disagreeing test logs (seeded models averaged)
    size = {}
    for name, arrs in models.items():
        size[name] = {k: float(np.mean([np.mean(np.abs(a[te & m])) for a in arrs]))
                      for k, m in (('all', np.ones(len(d), bool)), ('agree', d == 0), ('disagree', d != 0))}
    out.setdefault(proto, {})['prediction_size'] = size
    groups = {'all': te}
    for s in ROUTE_LABELS:
        groups[f'route history {s}'] = te & (rs == s)
    for s in CLIMBER_LABELS:
        groups[f'climber history {s}'] = te & (cs == s)
    groups['agrees with official'] = te & (d == 0); groups['disagrees'] = te & (d != 0)
    groups['softer'] = te & (d < 0); groups['harder'] = te & (d > 0)
    for new, ref in (('PACE', 'B1-RF'), ('PACE', 'B1-LR'), ('PACE-GBM', 'B3-GBM')):
        for gname, m in groups.items():
            if m.sum() < 200:
                continue
            bt = per_seed(lambda v: climber_bootstrap(d[m], {new: v[new][m], ref: v[ref][m]}, users[m], B=B, ref=ref),
                          {new: models[new], ref: models[ref]})
            hh = per_seed(lambda v: helped_hurt(d[m], v[new][m], v[ref][m], users[m], tol=1e-3), {new: models[new], ref: models[ref]}) if gname == 'all' else {}
            prow.append(dict(protocol=proto, new=new, ref=ref, group=gname, logs=int(m.sum()),
                             RMSE_new=bt[new]['RMSE'], RMSE_ref=bt[ref]['RMSE'], diff=bt[new]['diff'],
                             lo=bt[new]['diff_lo'], hi=bt[new]['diff_hi'],
                             climbers_better=hh.get('helped'), climbers_worse=hh.get('hurt')))
    print(proto, 'paired comparisons done', flush=True)
tag = '' if PROTOS == ['A', 'B'] else '_' + ''.join(PROTOS)
pd.DataFrame(prow).to_csv(f'{OUT_DIR}/paired_strata_indoor{tag}.csv', index=False)
json.dump(out, open(f'{OUT_DIR}/design_checks_indoor{tag}.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])
