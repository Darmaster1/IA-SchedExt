"""Run all experiments. Usage: python experiments/run_experiments.py [--quick]
Seeds: validation 1000-1007 (IA aging-parameter tuning only), test 0..N-1 (never used for tuning)."""
import sys, json, itertools, os, time
from dataclasses import replace, asdict
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from iasched.sim import Params, run_one

QUICK = "--quick" in sys.argv
N = 8 if QUICK else 50          # main runs per policy
M = 4 if QUICK else 20          # sweep runs
K = 4 if QUICK else 30          # ablation runs
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
POL = ["eevdf", "static", "lavd", "ia"]
BASE = Params(sim_ms=6000, warm_ms=1000, drain_ms=2000)
BURST = replace(BASE, burst_mult=2.0)
KEYS = ["p50", "p95", "p99", "p999", "slo_miss", "wait_p99", "bg_thpt", "bg_slow_med", "bg_slow_p99",
        "bg_dly_p99", "bg_dly_max", "util", "trans_per_s", "ovh_pct", "censored", "n_meas", "mode_frac"]


def runs(pol, p, seeds, keep=0):
    return [run_one((pol, p, s, s < keep)) for s in seeds]


def slim(rs):
    return [{k: r[k] for k in KEYS} | {"seed": r["seed"]} for r in rs]


def save(name, obj):
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, default=float)


t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


# ---- 0. tune IA aging parameters on validation seeds (bursty workload) ----
val = range(1000, 1008)
best = None
for boost, alpha, tg in itertools.product([2, 4, 8], [0.25, 0.5, 1.0], [500.0, 1000.0, 2000.0]):
    p = replace(BURST, boost=boost, alpha=alpha, t_gran=tg)
    rs = runs("ia", p, val)
    key = (float(np.median([r["p99"] for r in rs])), float(np.median([r["slo_miss"] for r in rs])))
    if best is None or key < best[0]:
        best = (key, dict(boost=boost, alpha=alpha, t_gran=tg))
TUNED = best[1]
save("tuned", {"params": TUNED, "val_p99_slo": best[0],
               "grid": "boost{2,4,8} x alpha{.25,.5,1} x t_gran{500,1000,2000}us; objective: median p99 then SLO miss"})
log(f"tuned {TUNED}")


def P(p, pol, **kw):
    return replace(p, **TUNED, **kw) if pol == "ia" else p


# ---- E1: main comparison (bursty and smooth mixed workload) ----
E1 = {}
for wname, wp in [("mixed_burst", BURST), ("mixed_smooth", BASE)]:
    E1[wname] = {}
    for pol in POL:
        rs = runs(pol, P(wp, pol), range(N), keep=10 if wname == "mixed_burst" else 0)
        if wname == "mixed_burst":
            np.savez_compressed(os.path.join(OUT, f"lat_{pol}.npz"),
                                lat=np.concatenate([np.array(r["lat"]) for r in rs if "lat" in r]))
        E1[wname][pol] = slim(rs)
    log(f"E1 {wname}")
save("e1_main", E1)

# ---- E2: single-class workloads A (pure FaaS) and B (pure background) ----
E2 = {}
for wname, wp in [("A_faas_only", replace(BASE, n_bg=0)), ("B_bg_only", replace(BASE, faas_rate=0.0))]:
    E2[wname] = {pol: slim(runs(pol, P(wp, pol), range(M))) for pol in POL}
log("E2"); save("e2_single", E2)

# ---- E3: sweeps over burst intensity and number of daemons ----
E3 = {"burst_mult": {}, "n_bg": {}}
for m in [1.0, 1.5, 2.0, 2.5]:
    E3["burst_mult"][str(m)] = {pol: slim(runs(pol, P(replace(BASE, burst_mult=m), pol), range(M))) for pol in POL}
for d in [1, 2, 4, 8]:
    E3["n_bg"][str(d)] = {pol: slim(runs(pol, P(replace(BURST, n_bg=d), pol), range(M))) for pol in POL}
log("E3"); save("e3_sweeps", E3)

# ---- E4: IA ablations and robustness (bursty workload) ----
variants = {
    "IA (full)": {},
    "no AGING (FAIR/STRICT)": dict(modes=(True, True, False)),
    "no STRICT (FAIR/AGING)": dict(modes=(True, False, True)),
    "no FAIR (STRICT/AGING)": dict(modes=(False, True, True)),
    "theta=0.01": dict(theta=0.01),
    "theta=0.20": dict(theta=0.20),
    "F thresholds 0.2/0.5": dict(f_lo=0.2, f_hi=0.5),
    "F thresholds 0.4/0.7": dict(f_lo=0.4, f_hi=0.7),
    "min dwell 5 ms": dict(dwell_min=5000.0),
    "misclassification 5%": dict(misclass=0.05),
    "misclassification 20%": dict(misclass=0.20),
}
E4 = {name: slim(runs("ia", P(BURST, "ia", **kw), range(K))) for name, kw in variants.items()}
log("E4"); save("e4_ablation", E4)

# ---- E5: sensitivity to interference-model strength gamma ----
E5 = {}
for g in [0.0, 0.15, 0.30, 0.60]:
    E5[str(g)] = {pol: slim(runs(pol, P(replace(BURST, gamma=g), pol), range(M))) for pol in ["eevdf", "static", "ia"]}
log("E5"); save("e5_gamma", E5)
save("meta", {"N": N, "M": M, "K": K, "base": asdict(BASE), "tuned": TUNED, "quick": QUICK})
log("done")
