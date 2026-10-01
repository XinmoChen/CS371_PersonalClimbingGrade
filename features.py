"""Extended prequential feature set for the gradient boosting baseline (B3). Every value uses only logs with a
strictly earlier timestamp. The set deliberately contains everything PACE sees (climber, route, setter and
their grade interactions) so that B3 tests whether an unstructured flexible learner can match PACE."""
import numpy as np
import pandas as pd
from common import prior_sum_count, codes


def _stats(prefix, key, t, dev, f):
    s, n = prior_sum_count(key, t, dev)
    dis, _ = prior_sum_count(key, t, (dev != 0).astype(float))
    hard, _ = prior_sum_count(key, t, (dev > 0).astype(float))
    soft, _ = prior_sum_count(key, t, (dev < 0).astype(float))
    safe = np.maximum(n, 1)
    f[f'{prefix}_n'] = n.astype(float)
    f[f'{prefix}_mean'] = np.where(n > 0, s / safe, 0.0)
    f[f'{prefix}_dis_rate'] = np.where(n > 0, dis / safe, np.nan)
    f[f'{prefix}_hard_rate'] = np.where(n > 0, hard / safe, np.nan)
    f[f'{prefix}_soft_rate'] = np.where(n > 0, soft / safe, np.nan)


def extended_features(df, andric_feats=None):
    t, dev = df.t.values, df.dev.values.astype(float)
    g = df.grade_id.values
    f = {'grade_id': g.astype(float)}
    _stats('route', df.route_id.values, t, dev, f)
    _stats('climber', df.user_id.values, t, dev, f)
    _stats('climber_grade', codes(df.user_id.values, g), t, dev, f)
    if 'route_setter_id' in df:
        _stats('setter', df.route_setter_id.values, t, dev, f)
        _stats('setter_grade', codes(df.route_setter_id.values, g), t, dev, f)
    else:
        for c in ['rock_type', 'steepness', 'length', 'altitude']:
            f[c] = pd.to_numeric(df[c], errors='coerce').values.astype(float)
    # climber level: mean official grade of earlier logs, and how far this route sits from it
    s, n = prior_sum_count(df.user_id.values, t, g.astype(float))
    lvl = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    f['climber_level'] = lvl
    f['rel_grade'] = g - lvl
    # route age: days since the route's first log (0 for a first log)
    first = df.groupby('route_id').t.transform('min').values
    f['route_age_days'] = (t - first) / 86400.0
    if andric_feats is not None:
        for k, v in andric_feats.items():
            if k != 'grade_id':
                f[f'andric_{k}'] = v
    return pd.DataFrame(f)
