import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from tqdm import tqdm

PATH = "data/raw/machine_usage.csv"
OUT = "data/processed/machine_profile.csv"
COLS = ["machine_id", "time_stamp", "cpu", "mem", "mem_gps", "mkpi", "net_in", "net_out", "disk_io"]
USE = ["machine_id", "time_stamp", "cpu", "mem", "net_in", "net_out", "disk_io"]
FEATS = ["cpu", "mem", "net_in", "net_out", "disk_io"]
N_MIN = 8 * 24 * 60          # 8 days in minutes
MAXM = 6000                  # upper bound on machine index

covered = np.zeros((MAXM, N_MIN), dtype=bool)
n_rows = np.zeros(MAXM, dtype=np.int64)
bad = {c: np.zeros(MAXM, dtype=np.int64) for c in FEATS}
ts_min, ts_max, total, out_of_range, max_id = 10**12, 0, 0, 0, 0

reader = pd.read_csv(PATH, header=None, names=COLS, usecols=USE,
                     chunksize=5_000_000, dtype={"machine_id": "string"})
for ch in tqdm(reader, unit="chunk"):
    idx = ch["machine_id"].str.slice(2).astype("int64").to_numpy()
    ts = ch["time_stamp"].to_numpy(dtype=np.int64)
    max_id = max(max_id, int(idx.max()))
    assert max_id < MAXM, "machine index exceeds MAXM"
    minute = ts // 60
    ok = (minute >= 0) & (minute < N_MIN)
    out_of_range += int((~ok).sum())
    covered[idx[ok], minute[ok]] = True
    n_rows += np.bincount(idx, minlength=MAXM)
    for c in FEATS:
        v = ch[c].to_numpy(dtype=float)
        m = np.isnan(v) | (v < 0) | (v > 100)
        bad[c] += np.bincount(idx[m], minlength=MAXM)
    ts_min, ts_max = min(ts_min, int(ts.min())), max(ts_max, int(ts.max()))
    total += len(ch)

present = np.flatnonzero(n_rows > 0)
prof = pd.DataFrame({
    "machine": ["m_%d" % i for i in present],
    "n_rows": n_rows[present],
    "coverage": covered[present].sum(axis=1) / N_MIN,
})
for c in FEATS:
    prof["bad_" + c] = bad[c][present]
prof.to_csv(OUT, index=False)

print("total rows:", total)
print("machines:", len(prof), " max machine index:", max_id)
print("time_stamp min/max (s):", ts_min, ts_max, " rows outside 8-day range:", out_of_range)
print("coverage quantiles:")
print(prof["coverage"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
for t in (0.99, 0.95, 0.90):
    print("machines with coverage >= %.2f: %d" % (t, int((prof["coverage"] >= t).sum())))
print(prof.sort_values("coverage", ascending=False).head(8).to_string(index=False))