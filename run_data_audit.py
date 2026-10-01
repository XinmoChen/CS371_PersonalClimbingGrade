"""Data facts quoted in the paper that no model computes. Output: results/data_audit.json.
- Smallest number of logs per climber in each released file.
- Protocol B indoor test logs with no earlier log by the same climber, and how many new climbers they come from.
- Share of indoor logs with the proposed grade flag set that disagree with the official grade. This is the ONLY
  place grade_proposal is read: it is read here to justify excluding it, and no model or feature ever loads it."""
import json, os
import numpy as np, pandas as pd
from common import DATA_DIR, OUT_DIR, load, split_protocol_b

out = {}
for setting in ('indoor', 'outdoor'):
    g = load(setting)
    out[f'min_logs_per_climber_{setting}'] = int(g.groupby('user_id').size().min())
g = load('indoor'); sp = split_protocol_b(g, 'indoor'); te = sp == 'test'
seen = set(g.user_id[sp != 'test'])
first = g.groupby('user_id').t.transform('min').values == g.t.values   # logs at a climber's first timestamp
no_prior = te & ~g.user_id.isin(seen).values & first
out['protocolB_indoor_test_logs'] = int(te.sum())
out['protocolB_indoor_test_logs_no_earlier_climber_log'] = int(no_prior.sum())
out['protocolB_indoor_new_climbers'] = int(g.user_id[te & ~g.user_id.isin(seen).values].nunique())
raw = pd.read_csv(os.path.join(DATA_DIR, 'gym_routes_raw.csv'), usecols=['grade_proposal', 'grade_id', 'user_grade_id'])
flag = raw.grade_proposal.fillna(0).astype(float) != 0
out['indoor_logs_with_grade_proposal_flag'] = int(flag.sum())
out['share_flagged_that_disagree'] = float((raw.user_grade_id[flag] != raw.grade_id[flag]).mean())
print(json.dumps(out, indent=1))
json.dump(out, open(f'{OUT_DIR}/data_audit.json', 'w'), indent=1)
