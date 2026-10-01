"""Shared data loading, prequential statistics, splits and strata for the Team 8 climbing project.

Conventions used everywhere in this code base
  * A log is one ascent record (climber i, route r, setter s, timestamp t, official grade g, perceived grade y).
  * The deviation d = y - g is an integer in [-3, 3].
  * "Prior" or "prequential" statistics for a log use only logs with a STRICTLY EARLIER timestamp,
    exactly like the released SQL of Andric et al. (2021) (`date < t`).
  * Logs are sorted by (timestamp, id) once, and every array in this code base follows that order.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Defaults: the released data cloned next to this folder, and results/ inside it. Both can be overridden.
DATA_DIR = os.environ.get('CLIMB_DATA', os.path.join(HERE, '..', 'climbing_grade_prediction', 'data'))
OUT_DIR = os.environ.get('PACE_OUT', os.path.join(HERE, 'results'))
LEAKY_COLUMNS = ['grade_proposal', 'rating', 'recommended', 'user_style', 'tries', 'repeats', 'sits', 'project',
                 'climb_type', 'user_climb_type', 'protection',
                 'user_athletic', 'user_cruxes', 'user_endurance', 'user_fingerstrength', 'user_sloppers',
                 'user_technical', 'user_body_power']
# Columns a pre-climb predictor may read. Everything else is excluded (see LEAKY_COLUMNS and the audit).
ALLOWED_INDOOR = ['id', 'user_id', 'route_id', 'route_setter_id', 'date', 'grade_id', 'user_grade_id', 'train']
ALLOWED_OUTDOOR = ['id', 'user_id', 'route_id', 'date', 'grade_id', 'user_grade_id', 'train',
                   'altitude', 'rock_type', 'length', 'steepness']


def load(setting='indoor'):
    """Load one setting, keep only allowed columns, sort by (date, id), add deviation and integer time keys."""
    fn = 'gym_routes_raw.csv' if setting == 'indoor' else 'routes_raw.csv'
    cols = ALLOWED_INDOOR if setting == 'indoor' else ALLOWED_OUTDOOR
    df = pd.read_csv(os.path.join(DATA_DIR, fn), usecols=cols, parse_dates=['date'])[cols]   # other columns are never read
    df = df.sort_values(['date', 'id'], kind='mergesort').reset_index(drop=True)
    df['dev'] = (df.user_grade_id - df.grade_id).astype(np.int64)
    df['t'] = df.date.values.astype('datetime64[s]').astype(np.int64)      # seconds, for strict comparisons
    df['day'] = df.date.dt.normalize().values.astype('datetime64[D]').astype(np.int64)
    df['setting'] = setting
    return df


def codes(*arrays):
    """Factorize one or several aligned key arrays into a single integer code per row."""
    if len(arrays) == 1:
        return pd.factorize(pd.Series(arrays[0]))[0].astype(np.int64)
    frame = pd.DataFrame({f'k{j}': a for j, a in enumerate(arrays)})
    return frame.groupby(list(frame.columns), sort=False).ngroup().values.astype(np.int64)


def prior_sum_count(key, t, values):
    """For every row, sum and count of `values` over rows with the same key and a strictly earlier time t.

    key, t, values are aligned 1-d arrays (any order). Ties in (key, t) share the same prefix, so a row never
    sees another row logged at the same timestamp (conservative, matches `date < t`)."""
    key = np.asarray(key); t = np.asarray(t); v = np.asarray(values, dtype=np.float64)
    order = np.lexsort((t, key))
    k, tt, vv = key[order], t[order], v[order]
    ex = np.cumsum(vv) - vv                                   # exclusive prefix over the whole array
    n = len(k)
    new_key = np.ones(n, bool); new_key[1:] = k[1:] != k[:-1]
    key_start = np.maximum.accumulate(np.where(new_key, np.arange(n), 0))
    ex = ex - ex[key_start]
    cnt = np.arange(n) - key_start
    new_tie = new_key.copy(); new_tie[1:] |= tt[1:] != tt[:-1]
    tie_start = np.maximum.accumulate(np.where(new_tie, np.arange(n), 0))
    ex = ex[tie_start]; cnt = cnt[tie_start]
    s = np.empty(n); c = np.empty(n, dtype=np.int64)
    s[order] = ex; c[order] = cnt
    return s, c


def prior_count(key, t):
    return prior_sum_count(key, t, np.zeros(len(key)))[1]


def add_history_counts(df):
    """Prior log counts per climber, route and setter (strictly earlier timestamp)."""
    df['n_climber'] = prior_count(df.user_id.values, df.t.values)
    df['n_route'] = prior_count(df.route_id.values, df.t.values)
    if 'route_setter_id' in df:
        df['n_setter'] = prior_count(df.route_setter_id.values, df.t.values)
    return df


# ---------------------------------------------------------------- splits
def split_protocol_a(df, dev_frac=0.10):
    """Protocol A: the published per-climber chronological 80/20 split (train flag of Andric et al.).
    The development set is the last 10% of each climber's training logs (chronological), used for tuning only."""
    split = np.where(df.train.values == 1, 'train', 'test').astype(object)
    tr = np.where(df.train.values == 1)[0]
    rank = df.iloc[tr].groupby('user_id').cumcount().values
    size = df.iloc[tr].groupby('user_id').user_id.transform('size').values
    is_dev = rank >= np.floor((1 - dev_frac) * size)
    split[tr[is_dev]] = 'dev'
    return split


PROTOCOL_B_CUTS = {'indoor': ('2019-07-01', '2019-09-01'), 'outdoor': ('2018-01-01', '2019-01-01')}


def split_protocol_b(df, setting):
    """Protocol B: one global forward split in time. Nothing from the future of a test log is used anywhere."""
    dev_start, test_start = [np.datetime64(x) for x in PROTOCOL_B_CUTS[setting]]
    d = df.date.values
    return np.where(d < dev_start, 'train', np.where(d < test_start, 'dev', 'test')).astype(object)


# ---------------------------------------------------------------- strata
ROUTE_BINS = [-1, 0, 2, 4, 9, 10 ** 9]
ROUTE_LABELS = ['0', '1-2', '3-4', '5-9', '10+']
CLIMBER_BINS = [-1, 4, 9, 19, 49, 10 ** 9]
CLIMBER_LABELS = ['0-4', '5-9', '10-19', '20-49', '50+']


def route_stratum(n):
    return pd.cut(n, ROUTE_BINS, labels=ROUTE_LABELS).astype(str)


def climber_stratum(n):
    return pd.cut(n, CLIMBER_BINS, labels=CLIMBER_LABELS).astype(str)
