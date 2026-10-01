"""Wall clock time of one full prequential pass over the indoor data (tuned Protocol A penalties), run alone on an
otherwise idle machine so the number is not inflated by other jobs. Output: results/timing_indoor.json."""
import json, time, os
import numpy as np
from common import *
from pace_model import Block, prequential

g = load('indoor'); d = g.dev.values.astype(float)
X = np.column_stack([np.ones(len(g)), g.grade_id.values - 13.0])
L = json.load(open(f'{OUT_DIR}/pace_lambdas_indoor.json'))
C = {'climber': codes(g.user_id.values), 'route': codes(g.route_id.values), 'env': codes(g.route_setter_id.values),
     'climber_grade': codes(g.user_id.values, g.grade_id.values), 'env_grade': codes(g.route_setter_id.values, g.grade_id.values)}
out = {'load_average_before': os.getloadavg()[0]}
for name, ks in (('PACE', ['climber', 'route', 'env']), ('PACE-X', ['climber', 'route', 'env', 'climber_grade', 'env_grade'])):
    t0 = time.time()
    prequential(d, X, [Block(k, C[k], L[k]) for k in ks], g.day.values, g.t.values)
    out[name] = round(time.time() - t0, 1)
    print(name, out[name], flush=True)
out['days'] = int(len(np.unique(g.day.values))); out['logs'] = int(len(g))
json.dump(out, open(f'{OUT_DIR}/timing_indoor.json', 'w'), indent=1)
