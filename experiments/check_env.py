import sys
sys.path.insert(0, ".")

import numpy, pandas, sklearn, torch, statsmodels, matplotlib, networkx, yaml, tqdm
from src import set_seed, load_config

cfg = load_config()
set_seed(cfg["seeds"][0])

print("python", sys.version.split()[0])
for m in (numpy, pandas, sklearn, torch, statsmodels, matplotlib, networkx, yaml):
    print(m.__name__, m.__version__)
print("cuda available:", torch.cuda.is_available())
print("config keys:", list(cfg))
print("OK")