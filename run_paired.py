"""Paired climber cluster bootstrap for the comparisons quoted in the text.
Seeded models are compared seed by seed (seed s against seed s, or against the single deterministic model), and the
difference and its interval endpoints are averaged over the five seeds, matching the seed means of the main table."""
import json, sys
import numpy as np
from common import *
from metrics import climber_bootstrap, per_seed, rounded

out = {}
for setting in sys.argv[1:] or ['indoor']:
    g = load(setting); d = g.dev.values.astype(float)
    for proto, sp in (('A', split_protocol_a(g)), ('B', split_protocol_b(g, setting))):
        te = sp == 'test'
        P = np.load(f'{OUT_DIR}/pace_preds_{setting}' + ('' if proto == 'A' else '_B') + '.npz')
        B = dict(np.load(f'{OUT_DIR}/baseline_preds_{setting}.npz')); B.update(dict(np.load(f'{OUT_DIR}/gbm_variant_preds_{setting}.npz')))
        seeds = lambda name: [B[k].astype(float) for k in sorted(B) if k.startswith(f'{proto}|{name}|')]
        preds = {'PACE': [P['PACE|pred']], 'PACE-noU': [P['PACE|pred'] - P['PACE|val|climber']], 'PACE-X': [P['PACE-X|pred']],
                 'C': [P['PACE[route+env]|pred']], 'B0': [np.zeros(len(g))], 'B1-RF': seeds('B1-RF'), 'B1-LR': seeds('B1-LR'), 'B3-GBM': seeds('B3-GBM'),
                 'B3-GBM[-climber]': seeds('B3-GBM[-climber]'), 'PACE-GBM': seeds('PACE-GBM'), 'PACE-GBM[-climber]': seeds('PACE-GBM[-climber]')}
        preds = {k: [a[te] for a in v] for k, v in preds.items() if v}
        users = g.user_id.values[te]; y = d[te]
        pairs = [('PACE', 'PACE-noU'), ('PACE', 'C'), ('PACE', 'B1-RF'), ('PACE-X', 'B1-RF'), ('PACE-GBM', 'B3-GBM'), ('PACE-GBM', 'PACE'),
                 ('B3-GBM', 'B3-GBM[-climber]'), ('PACE-GBM', 'PACE-GBM[-climber]'), ('B3-GBM', 'B1-RF'), ('B3-GBM', 'PACE'),
                 ('PACE', 'B0'), ('B1-RF', 'B0'), ('B3-GBM', 'B0'), ('PACE-GBM', 'B0')]
        # average climber error (uRMSE, the second published metric) on displayed predictions, paired by climber
        uc, inv = np.unique(users, return_inverse=True); nu = len(uc); cnt = np.bincount(inv).astype(float)
        W = np.random.default_rng(0).multinomial(nu, np.full(nu, 1.0 / nu), size=1000).astype(float)
        def urmse_pair(pa, pb):
            ra = np.sqrt(np.bincount(inv, (y - rounded(pa)) ** 2) / cnt); rb = np.sqrt(np.bincount(inv, (y - rounded(pb)) ** 2) / cnt)
            bt = W @ (ra - rb) / nu
            return dict(uRMSE=float(ra.mean()), uRMSE_ref=float(rb.mean()), diff=float(ra.mean() - rb.mean()),
                        diff_lo=float(np.percentile(bt, 2.5)), diff_hi=float(np.percentile(bt, 97.5)))
        for a, b in (('PACE', 'B1-RF'), ('PACE', 'B1-LR'), ('PACE-GBM', 'B3-GBM'), ('PACE-GBM', 'B1-RF')):
            if a in preds and b in preds:
                r = per_seed(lambda v: urmse_pair(v[a], v[b]), {a: preds[a], b: preds[b]})
                out[f'{setting}|{proto}|uRMSE {a} vs {b}'] = {k: round(v, 5) for k, v in r.items()}
                print(setting, proto, 'uRMSE', a, 'vs', b, {k: round(v, 4) for k, v in r.items()}, flush=True)
        for a, b in pairs:
            if a not in preds or b not in preds:
                continue
            r = per_seed(lambda v: climber_bootstrap(y, {a: v[a], b: v[b]}, users, B=1000, ref=b)[a], {a: preds[a], b: preds[b]})
            out[f'{setting}|{proto}|{a} vs {b}'] = {k: round(v, 5) for k, v in r.items()}
            print(setting, proto, f'{a:22s} vs {b:20s}', {k: round(v, 4) for k, v in r.items()}, flush=True)
json.dump(out, open(f'{OUT_DIR}/paired_bootstrap.json', 'w'), indent=1)
