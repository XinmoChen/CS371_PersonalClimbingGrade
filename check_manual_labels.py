"""Agreement between the rule based error causes (run_errors.py) and an independent reading of the 100 sampled errors
(results/errors_manual_labels.csv, one label per sampled error, written while looking only at each error's route and
climber history, not at the rule label). Output: results/errors_manual_check.json."""
import json
import pandas as pd
from common import OUT_DIR
s = pd.read_csv(f'{OUT_DIR}/errors_sample.csv'); m = pd.read_csv(f'{OUT_DIR}/errors_manual_labels.csv')
df = s.merge(m, on='row')
agree = df.cause == df.manual_label
pairs = df[~agree].groupby(['cause', 'manual_label']).size().sort_values(ascending=False)
out = {'n': int(len(df)), 'agree': int(agree.sum()),
       'disagreements': [dict(rule=a, reading=b, count=int(c)) for (a, b), c in pairs.items()],
       'reading_distribution': df.manual_label.value_counts().to_dict()}
print(json.dumps(out, indent=1))
json.dump(out, open(f'{OUT_DIR}/errors_manual_check.json', 'w'), indent=1)
