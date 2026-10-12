import numpy as np

arr = np.load("data/processed/usage_1min.npy", mmap_mode="r")    # [F, M, T]
pres = ~np.isnan(np.asarray(arr[0]))                               # cpu, [M, T]
active = pres.any(axis=1)
frac = pres[active].mean(axis=0)                                   # share reporting per minute
good = frac >= 0.95

d = np.diff(np.concatenate(([0], good.astype(np.int8), [0])))
starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
lens = ends - starts
order = np.argsort(-lens)
print("good minutes (>=95%% of machines): %d of %d" % (good.sum(), len(good)))
print("segments:", len(lens))
print("top 10 segments: start_min, end_min, length_min, length_hours, start_day")
for i in order[:10]:
    print(starts[i], ends[i], lens[i], round(lens[i] / 60, 1), round(starts[i] / 1440, 2))
print("segments >= 360 min:", int((lens >= 360).sum()),
      " total minutes in them:", int(lens[lens >= 360].sum()))