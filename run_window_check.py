"""Outdoor diagnostic for Section 5.5: does recent route evidence help? Linear regression on the released outdoor
SSE features with the lifetime route mean only, against the same model with the 1.5 year windowed route means added.
Protocol B refits at the start of every test month. Output: results/window_check_outdoor.json."""
import numpy as np, pandas as pd, sys, json

from common import *
from andric import outdoor_features
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
g = load('outdoor'); AF = pd.DataFrame(outdoor_features(g)); y = g.user_grade_id.values.astype(float); gr = g.grade_id.values; d = g.dev.values.astype(float)
res = {}
for proto, sp in (('A', split_protocol_a(g)), ('B', split_protocol_b(g, 'outdoor'))):
    fit = np.isin(sp, ['train', 'dev']); te = sp == 'test'
    for name, cols in (('grade + lifetime route mean', ['grade_id', 'route_dev_sign']),
                       ('grade + lifetime and 1.5 year route means', ['grade_id', 'route_dev_sign', 'route_dev_sign_1year', 'route_dev_sign_1year_season'])):
        if proto == 'B':   # monthly refits, as in the baselines
            month = g.date.dt.to_period('M').values; p = np.full(len(g), np.nan)
            for m in pd.unique(month[te]):
                pm = np.where(te & (month == m))[0]; fi = np.arange(0, pm.min())
                sc = StandardScaler().fit(AF.iloc[fi][cols]); lr = LinearRegression().fit(sc.transform(AF.iloc[fi][cols]), y[fi])
                p[pm] = lr.predict(sc.transform(AF.iloc[pm][cols])) - gr[pm]
        else:
            sc = StandardScaler().fit(AF[fit][cols]); lr = LinearRegression().fit(sc.transform(AF[fit][cols]), y[fit])
            p = lr.predict(sc.transform(AF[cols])) - gr
        res[f'{proto}|{name}'] = float(np.sqrt(np.mean((d[te] - p[te]) ** 2)))
        print(proto, name, round(res[f'{proto}|{name}'], 4), flush=True)
json.dump(res, open(f'{OUT_DIR}/window_check_outdoor.json', 'w'), indent=1)
