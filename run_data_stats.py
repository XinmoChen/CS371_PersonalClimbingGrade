"""Dataset and split statistics for the Experiments section (Tables of data and splits)."""
import json
import numpy as np, pandas as pd
from common import *

rows = []
for setting in ('indoor', 'outdoor'):
    g = add_history_counts(load(setting))
    for proto, sp in (('A', split_protocol_a(g)), ('B', split_protocol_b(g, setting))):
        for part in ('train', 'dev', 'test'):
            m = sp == part
            s = g[m]
            seen_climbers = set(g.user_id[g.t < s.t.min()]) if part != 'train' else set()
            rows.append(dict(setting=setting, protocol=proto, split=part, logs=int(m.sum()),
                             climbers=s.user_id.nunique(), routes=s.route_id.nunique(),
                             setters=s.route_setter_id.nunique() if 'route_setter_id' in s else np.nan,
                             date_min=str(s.date.min().date()), date_max=str(s.date.max().date()),
                             disagree=float((s.dev != 0).mean()), softer=float((s.dev < 0).mean()), harder=float((s.dev > 0).mean()),
                             new_route=float((s.n_route == 0).mean()), new_climber_logs=float((s.n_climber == 0).mean()),
                             climber_hist_median=float(s.n_climber.median()), route_hist_median=float(s.n_route.median())))
    # whole-set facts
    life = (g.groupby('route_id').date.max() - g.groupby('route_id').date.min()).dt.days
    facts = dict(setting=setting, logs=len(g), climbers=g.user_id.nunique(), routes=g.route_id.nunique(),
                 setters=int(g.route_setter_id.nunique()) if 'route_setter_id' in g else None,
                 date_min=str(g.date.min().date()), date_max=str(g.date.max().date()),
                 disagree=float((g.dev != 0).mean()), dev_counts={int(k): int(v) for k, v in g.dev.value_counts().sort_index().items()},
                 route_life_median_days=float(life.median()), first_log_share=float((g.n_route == 0).mean()),
                 climbers_disagree_once=float(g.assign(x=g.dev != 0).groupby('user_id').x.any().mean()),
                 logs_per_climber_median=float(g.groupby('user_id').size().median()),
                 logs_per_route_median=float(g.groupby('route_id').size().median()),
                 grade_min=int(g.grade_id.min()), grade_max=int(g.grade_id.max()))
    json.dump(facts, open(f'{OUT_DIR}/data_facts_{setting}.json', 'w'), indent=1)
    print(json.dumps(facts))
df = pd.DataFrame(rows)
df.to_csv(f'{OUT_DIR}/split_stats.csv', index=False)
print(df.round(3).to_string())
# leakage checks on the published split used by static models (B2 SVD): future route information
g = load('indoor'); tr = g.train.values == 1
last_train_t = g[tr].groupby('route_id').t.max()
te = g[~tr]
share = float((te.t.values < te.route_id.map(last_train_t).fillna(-1).values).mean())
print('share of Protocol A test logs whose route has a training log dated later:', round(share, 4))
json.dump({'protoA_test_logs_with_later_train_logs_same_route': share}, open(f'{OUT_DIR}/leak_check.json', 'w'))
