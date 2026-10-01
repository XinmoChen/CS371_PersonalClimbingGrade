"""Records the software versions and hardware the results were produced on. Output: results/environment.json."""
import json, os, platform, re
import numpy, pandas, sklearn, lightgbm, statsmodels
try:
    import surprise; sv = surprise.__version__
except Exception:
    sv = 'unavailable'
from common import OUT_DIR
cpu = open('/proc/cpuinfo').read(); mem = open('/proc/meminfo').read()
out = {'python': platform.python_version(), 'numpy': numpy.__version__, 'pandas': pandas.__version__, 'scikit-learn': sklearn.__version__,
       'lightgbm': lightgbm.__version__, 'statsmodels': statsmodels.__version__, 'surprise': sv,
       'cpu_model': re.search(r'model name\s*:\s*(.*)', cpu).group(1), 'cpu_cores': os.cpu_count(),
       'memory_gb': round(int(re.search(r'MemTotal:\s*(\d+)', mem).group(1)) / 1024 ** 2, 1)}
print(json.dumps(out, indent=1)); json.dump(out, open(f'{OUT_DIR}/environment.json', 'w'), indent=1)
