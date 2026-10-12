"""Phase 2: Alibaba 2018 machine_usage -> [T, N, F] tensor + mask + Abilene topology."""
import numpy as np
import pandas as pd
import yaml

FEATS = ["cpu", "mem", "net_in", "net_out", "disk_io"]
ARR_PATH = "data/processed/usage_1min.npy"      # [F, machine, minute]
WINDOW = (2679, 9352)                           # contiguous stretch found by experiments/segments.py

NODES = ["ATLAM5", "ATLAng", "CHINng", "DNVRng", "HSTNng", "IPLSng",
         "KSCYng", "LOSAng", "NYCMng", "SNVAng", "STTLng", "WASHng"]
_EDGE_NAMES = [("ATLAM5", "ATLAng"), ("ATLAng", "HSTNng"), ("ATLAng", "WASHng"),
               ("ATLAng", "IPLSng"), ("CHINng", "IPLSng"), ("CHINng", "NYCMng"),
               ("DNVRng", "KSCYng"), ("DNVRng", "SNVAng"), ("DNVRng", "STTLng"),
               ("HSTNng", "KSCYng"), ("HSTNng", "LOSAng"), ("IPLSng", "KSCYng"),
               ("LOSAng", "SNVAng"), ("NYCMng", "WASHng"), ("SNVAng", "STTLng")]
EDGES = np.array([(NODES.index(a), NODES.index(b)) for a, b in _EDGE_NAMES])


def load_cfg(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def fill_short_gaps(x, max_gap):
    s = pd.Series(x)
    isn = s.isna()
    run_id = (isn != isn.shift()).cumsum()
    run_len = isn.groupby(run_id).transform("sum")
    filled = s.interpolate(limit_area="inside")
    filled[isn & (run_len > max_gap)] = np.nan
    return filled.to_numpy()


def select_machines(win, n, seed, min_cov):
    """win: [F, M, T] window. Candidates: coverage >= min_cov in EVERY feature inside the window."""
    cov = np.ones(win.shape[1])
    for k in range(win.shape[0]):
        cov = np.minimum(cov, (~np.isnan(win[k])).mean(axis=1))
    cand = np.flatnonzero(cov >= min_cov)
    if len(cand) < n:
        raise ValueError("only %d candidates at min_cov=%.2f" % (len(cand), min_cov))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(cand, size=n, replace=False)), len(cand)


def build_tensor(cfg=None, machine_seed=0, min_cov=0.97, max_gap=5):
    cfg = cfg or load_cfg()
    n = cfg["data"]["n_nodes"]
    arr = np.load(ARR_PATH, mmap_mode="r")
    win = np.asarray(arr[:, :, WINDOW[0]:WINDOW[1]], dtype=np.float32)
    ids, n_cand = select_machines(win, n, machine_seed, min_cov)
    x = win[:, ids, :].copy()
    for k in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[k, j] = fill_short_gaps(x[k, j], max_gap)
    x = np.transpose(x, (2, 1, 0))                            # [T, N, F]
    mask = ~np.isnan(x)
    T = x.shape[0]
    s = cfg["data"]["split"]
    tr_f, va_f = (s[0], s[1]) if isinstance(s, list) else (s["train"], s["val"])
    i1 = int(T * tr_f)
    i2 = i1 + int(T * va_f)
    splits = {"train": (0, i1), "val": (i1, i2), "test": (i2, T)}
    tr, trm = x[:i1], mask[:i1]
    mu = np.array([tr[..., k][trm[..., k]].mean() for k in range(x.shape[2])], dtype=np.float32)
    sd = np.array([tr[..., k][trm[..., k]].std() + 1e-6 for k in range(x.shape[2])], dtype=np.float32)
    xn = np.where(mask, (x - mu) / sd, 0.0).astype(np.float32)
    return dict(x=x, xn=xn, mask=mask, mu=mu, sd=sd, splits=splits,
                machine_ids=ids, n_candidates=n_cand)


def build_topology(cfg=None, seed=0):
    rng = np.random.default_rng(seed)
    base_delay = rng.uniform(2.0, 15.0, size=len(EDGES)).astype(np.float32)
    adj = np.zeros((len(NODES), len(NODES)), dtype=np.float32)
    for (a, b) in EDGES:
        adj[a, b] = adj[b, a] = 1.0
    return dict(nodes=NODES, edges=EDGES, base_delay=base_delay, adj=adj)


def link_load(x, edges):
    net = 0.5 * (x[..., 2] + x[..., 3])
    return 0.5 * (net[:, edges[:, 0]] + net[:, edges[:, 1]])


if __name__ == "__main__":
    d = build_tensor()
    x, mask = d["x"], d["mask"]
    print("window (min):", WINDOW, " length:", WINDOW[1] - WINDOW[0])
    print("candidates (cov>=0.97 in window, all features):", d["n_candidates"])
    print("machine ids:", ["m_%d" % i for i in d["machine_ids"]])
    print("tensor [T,N,F]:", x.shape, " splits:", d["splits"])
    print("valid fraction per feature:", mask.mean(axis=(0, 1)).round(4))
    print("valid fraction per node:", mask.all(axis=2).mean(axis=0).round(4))
    allv = mask.all(axis=(1, 2))
    print("minutes with all nodes+features valid:", round(float(allv.mean()), 4))
    bad = ~allv
    dd = np.diff(np.concatenate(([0], bad.astype(np.int8), [0])))
    runs = np.flatnonzero(dd == -1) - np.flatnonzero(dd == 1)
    print("invalid runs:", len(runs), " longest (min):", int(runs.max()) if len(runs) else 0)
    for name, (a, b) in d["splits"].items():
        print(name, "valid frac:", round(float(mask[a:b].mean()), 4))
    topo = build_topology()
    print("edges:", len(topo["edges"]), " base delay ms:", topo["base_delay"].round(1))
    ll = link_load(np.nan_to_num(x), topo["edges"])
    print("link load [T,E]:", ll.shape, " mean:", round(float(ll.mean()), 2))
    np.savez_compressed("data/processed/tensor_seed0.npz", xn=d["xn"], mask=mask, mu=d["mu"],
                        sd=d["sd"], machine_ids=d["machine_ids"])