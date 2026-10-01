"""Re-implementation of the knowledge based features of Andric, Ivanova and Ricci (2021) in pandas/numpy.

Follows the released SQL (Knowledge-based Regression/*/..._sql_queries.py) and the released configuration:
indoor uses grade_range = 0 and shrinkage k = 4, outdoor uses GRADE_OFFSET = 1 and shrinkage k = 4.
Every statistic uses logs with a strictly earlier timestamp (the SQL filter `date < t`).
"""
import numpy as np
from common import prior_sum_count, codes

K = 4.0  # shrinkage constant of the released configuration


def shrunk(key, t, dev, k=K):
    s, n = prior_sum_count(key, t, dev)
    return np.where(n > 0, s / (n + k), 0.0), n


def banded_user_dev(df, offset, k=K, default=0.0, extra_mask=None):
    """Climber deviation on routes of the same grade; if none, on grades within +-offset; else `default`.
    Mirrors the CASE WHEN chain of sql_user_features."""
    u, t, g, dev = df.user_id.values, df.t.values, df.grade_id.values, df.dev.values.astype(float)
    same, n_same = shrunk(codes(u, g), t, dev, k)
    if offset == 0:
        return np.where(n_same > 0, same, default)
    # grades within +-offset: sum over the band = sum of per-grade prior sums for g-offset..g+offset
    tot_s = np.zeros(len(df)); tot_n = np.zeros(len(df))
    for delta in range(-offset, offset + 1):
        # prior logs of climber u at grade g+delta before t: build by querying a shifted key
        s, n = _prior_at_grade(u, t, g, dev, delta)
        tot_s += s; tot_n += n
    band = np.where(tot_n > 0, tot_s / (tot_n + k), default)
    return np.where(n_same > 0, same, band)


def _prior_at_grade(u, t, g, dev, delta):
    """Sum/count of the climber's earlier deviations on routes graded g+delta (query key differs from data key)."""
    import pandas as pd
    # events: data rows contribute to key (u, g); queries ask key (u, g+delta). Merge via one combined array.
    n = len(u)
    key_data = pd.MultiIndex.from_arrays([u, g])
    key_query = pd.MultiIndex.from_arrays([u, g + delta])
    allkeys = key_data.append(key_query)
    kc = pd.factorize(allkeys)[0]
    kd, kq = kc[:n], kc[n:]
    # concatenate data events (value dev) and query events (value 0); a query at time t sees data with time < t
    key = np.concatenate([kd, kq]); tt = np.concatenate([t, t]); vv = np.concatenate([dev, np.zeros(n)])
    is_query = np.concatenate([np.zeros(n, bool), np.ones(n, bool)])
    s, c = prior_sum_count_mixed(key, tt, vv, is_query)
    return s[n:], c[n:]


def prior_sum_count_mixed(key, t, v, is_query):
    """Like prior_sum_count but only non-query rows contribute to sums and counts."""
    s, _ = prior_sum_count(key, t, np.where(is_query, 0.0, v))
    c_s, _ = prior_sum_count(key, t, np.where(is_query, 0.0, 1.0))
    return s, c_s.astype(np.int64)


def indoor_features(df):
    t, dev = df.t.values, df.dev.values.astype(float)
    f = {}
    f['grade_id'] = df.grade_id.values.astype(float)
    f['route_dev_sign'], _ = shrunk(df.route_id.values, t, dev)
    f['user_dev_sign'] = banded_user_dev(df, offset=0)
    s_same, n_same = shrunk(codes(df.route_setter_id.values, df.grade_id.values), t, dev)
    f['rs_dev_sign'] = np.where(n_same > 0, s_same, 0.0)
    return f


def mad_prior(key, t, dev, window=None, day=None):
    """Mean absolute deviation of earlier logs for `key` (optionally within a trailing window in days)."""
    s, n = prior_sum_count(key, t, np.abs(dev))
    if window is not None:
        s2, n2 = _windowed(key, t, np.abs(dev), window)
        s, n = s2, n2
    return np.where(n > 0, s / np.maximum(n, 1), 0.0)


def _windowed(key, t, v, window_sec):
    """Sum/count over earlier rows of the same key with t - window <= t' < t (two prefix queries)."""
    import pandas as pd
    s_all, n_all = prior_sum_count(key, t, v)
    n = len(key)
    # prefix strictly before (t - window): query rows at time t - window
    kk = np.concatenate([key, key]); tt = np.concatenate([t, t - window_sec]); vv = np.concatenate([v, np.zeros(n)])
    isq = np.concatenate([np.zeros(n, bool), np.ones(n, bool)])
    s_old, n_old = prior_sum_count_mixed(kk, tt, vv, isq)
    return s_all - s_old[n:], n_all - n_old[n:]


def outdoor_features(df):
    """Outdoor feature set named in the released configuration (routes_user_grade_prediction_evaluation.py).
    The released SQL computes the *_mad features with the mad_* functions (mean absolute deviation, grade +-1)."""
    import pandas as pd
    t, dev = df.t.values, df.dev.values.astype(float)
    month = df.date.dt.month.values
    f = {}
    f['grade_id'] = df.grade_id.values.astype(float)
    r = df.route_id.values
    f['route_dev_sign'], _ = shrunk(r, t, dev)
    win = int(1.5 * 365.25 * 86400)
    s, n = _windowed(r, t, dev, win)
    f['route_dev_sign_1year'] = np.where(n > 0, s / (n + K), 0.0)
    # 1.5 year window restricted to months within +-1 of the current month (SQL: month between m-1 and m+1)
    tot_s = np.zeros(len(df)); tot_n = np.zeros(len(df))
    for dm in (-1, 0, 1):
        # data key (route, month), query key (route, month+dm); no wrap around, as in the SQL
        kd = pd.MultiIndex.from_arrays([r, month]); kq = pd.MultiIndex.from_arrays([r, month + dm])
        kc = pd.factorize(kd.append(kq))[0]; N = len(df)
        kk = np.concatenate([kc[:N], kc[N:]]); tt = np.concatenate([t, t]); vv = np.concatenate([dev, np.zeros(N)])
        isq = np.concatenate([np.zeros(N, bool), np.ones(N, bool)])
        s_all, n_all = prior_sum_count_mixed(kk, tt, vv, isq)
        tt2 = np.concatenate([t, t - win])
        s_old, n_old = prior_sum_count_mixed(kk, tt2, vv, isq)
        tot_s += s_all[N:] - s_old[N:]; tot_n += n_all[N:] - n_old[N:]
    f['route_dev_sign_1year_season'] = np.where(tot_n > 0, tot_s / (tot_n + K), 0.0)
    f['route_grade_mad'] = mad_prior(r, t, dev)
    s, n = _windowed(r, t, np.abs(dev), win)
    f['route_grade_mad_1year'] = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    f['user_dev_sign'] = banded_user_dev(df, offset=1, default=0.09)
    # user MAD over grades within +-1 of the current grade
    u, g = df.user_id.values, df.grade_id.values
    tot_s = np.zeros(len(df)); tot_n = np.zeros(len(df))
    for delta in (-1, 0, 1):
        s, n = _prior_at_grade(u, t, g, np.abs(dev), delta)
        tot_s += s; tot_n += n
    f['user_grade_mad'] = np.where(tot_n > 0, tot_s / np.maximum(tot_n, 1), 0.0)
    f['season'] = np.select([np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8]), np.isin(month, [9, 10, 11])], [1, 2, 3], 4).astype(float)
    f['month'] = month.astype(float)
    f['year'] = df.date.dt.year.values.astype(float)
    for c in ['altitude', 'length', 'rock_type']:
        f[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).values.astype(float)
    return f
