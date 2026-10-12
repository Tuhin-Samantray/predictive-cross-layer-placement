import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from tqdm import tqdm

PATH = "data/raw/machine_usage.csv"
OUT = "data/processed/usage_1min.npy"
COLS = ["machine_id", "time_stamp", "cpu", "mem", "mem_gps", "mkpi", "net_in", "net_out", "disk_io"]
USE = ["machine_id", "time_stamp", "cpu", "mem", "net_in", "net_out", "disk_io"]
FEATS = ["cpu", "mem", "net_in", "net_out", "disk_io"]
N_MIN = 8 * 24 * 60
M = 4035                                   # machine ids m_1 .. m_4034
size = M * N_MIN

sums = np.zeros((len(FEATS), size), dtype=np.float32)
cnts = np.zeros((len(FEATS), size), dtype=np.uint16)

reader = pd.read_csv(PATH, header=None, names=COLS, usecols=USE,
                     chunksize=5_000_000, dtype={"machine_id": "string"})
for ch in tqdm(reader, unit="chunk"):
    idx = ch["machine_id"].str.slice(2).astype("int64").to_numpy()
    assert idx.max() < M
    minute = ch["time_stamp"].to_numpy(dtype=np.int64) // 60
    flat = idx * N_MIN + minute
    for k, c in enumerate(FEATS):
        v = ch[c].to_numpy(dtype=float)
        ok = ~np.isnan(v) & (v >= 0) & (v <= 100)      # drop NaN and -1/101 markers
        f = flat[ok]
        sums[k] += np.bincount(f, weights=v[ok], minlength=size).astype(np.float32)
        cnts[k] += np.bincount(f, minlength=size).astype(np.uint16)

print("max samples in one (machine, minute, feature):", int(cnts.max()))
for k in range(len(FEATS)):                              # sums -> means, NaN where no sample
    m = cnts[k] > 0
    sums[k][m] /= cnts[k][m]
    sums[k][~m] = np.nan
arr = sums.reshape(len(FEATS), M, N_MIN)                 # layout [F, machine, minute]
np.save(OUT, arr)

pres = ~np.isnan(arr[0])                                 # cpu present, [M, T]
active = pres.any(axis=1)
pres = pres[active]
ids = np.flatnonzero(active)
frac = pres.mean(axis=0)                                 # share of machines reporting each minute
print("\nmachines with data:", int(active.sum()))
print("share of machines reporting per minute:")
print(pd.Series(frac).describe(percentiles=[.01, .05, .1, .5]).round(3).to_string())
for t in (0.5, 0.9, 0.95):
    print("minutes with < %.2f of machines reporting: %d of %d" % (t, int((frac < t).sum()), N_MIN))
w = np.argsort(frac)[:8]
print("worst minutes (minute, day, share):", [(int(i), round(i / 1440, 2), round(float(frac[i]), 3)) for i in w])

def runs(p):
    miss = (~p).astype(np.int8)
    d = np.diff(np.concatenate(([0], miss, [0])))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)

cov = pres.mean(axis=1)
print("\ntop machines by coverage: id, coverage, n_gaps, longest_gap_min, median_gap_min")
for j in np.argsort(-cov)[:8]:
    r = runs(pres[j])
    print("m_%d" % ids[j], round(float(cov[j]), 3), len(r), int(r.max()) if len(r) else 0,
          float(np.median(r)) if len(r) else 0)
