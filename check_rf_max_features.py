"""Outdoor random forest with the released max_features = 4 (Appendix A, Baselines). The released configuration of
Andric et al. offers 4 features per split. Indoors that is every feature. Outdoors the released set has 14 features,
and with 4 per split the forest does not fit the absolute grade target. This script fits the forest of run_baselines.py
(500 trees, depth 7, 30% subsamples, seed 0) on training rows with 4 features per split and with every feature, and
scores development rows under both protocols. Output: results/rf_max_features_outdoor.json."""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from common import *
from andric import outdoor_features

g = load('outdoor')
d = g.dev.values.astype(float)
y_abs = g.user_grade_id.values.astype(float)
grade = g.grade_id.values.astype(float)
AF = pd.DataFrame(outdoor_features(g)); cols = list(AF.columns)
out = {'n_features': len(cols), 'seed': 0}
for proto, sp in (('A', split_protocol_a(g)), ('B', split_protocol_b(g, 'outdoor'))):
    tr = np.where(sp == 'train')[0]; dv = np.where(sp == 'dev')[0]
    sc = StandardScaler().fit(AF.iloc[tr][cols])
    for mf, name in ((4, 'released, 4 features per split'), (None, 'every feature per split')):
        m = RandomForestRegressor(n_estimators=500, max_depth=7, max_features=mf, max_samples=0.3, random_state=0, n_jobs=2)
        m.fit(sc.transform(AF.iloc[tr][cols]), y_abs[tr])
        p = m.predict(sc.transform(AF.iloc[dv][cols])) - grade[dv]
        out[f'{proto}|{name}|devRMSE'] = float(np.sqrt(np.mean((d[dv] - p) ** 2)))
        print(proto, name, round(out[f'{proto}|{name}|devRMSE'], 4), flush=True)
json.dump(out, open(f'{OUT_DIR}/rf_max_features_outdoor.json', 'w'), indent=1)
