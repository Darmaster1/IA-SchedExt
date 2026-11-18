"""Build LaTeX tables, figures and numbers.json from results/*.json (no numbers are typed by hand)."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIG, exist_ok=True)
L = lambda n: json.load(open(os.path.join(R, n + ".json")))
e1, e2, e3, e4, e5, meta, tuned = L("e1_main"), L("e2_single"), L("e3_sweeps"), L("e4_ablation"), L("e5_gamma"), L("meta"), L("tuned")
NAME = {"eevdf": "EEVDF", "static": "Static FaaS Prio.", "lavd": "LAVD-like", "ia": "IA-SchedExt"}
POL = ["eevdf", "static", "lavd", "ia"]
rng = np.random.default_rng(0)
med = lambda rs, k: float(np.median([r[k] for r in rs]))
q1 = lambda rs, k: float(np.percentile([r[k] for r in rs], 25)); q3 = lambda rs, k: float(np.percentile([r[k] for r in rs], 75))
num = {}

def boot_ci(x, B=2000):
    x = np.asarray(x); m = [np.median(rng.choice(x, len(x))) for _ in range(B)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def paired(wl, base, key="p99"):
    a = np.array([r[key] for r in e1[wl]["ia"]]); b = np.array([r[key] for r in e1[wl][base]])
    d = 100.0 * (a - b) / b
    p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
    lo, hi = boot_ci(d)
    return float(np.median(d)), lo, hi, float(p)

# ---------- main table ----------
rows = []
for wl, lab in [("mixed_burst", "Bursty mixed"), ("mixed_smooth", "Smooth mixed")]:
    for i, pol in enumerate(POL):
        rs = e1[wl][pol]
        rows.append("%s & %s & %.2f & %.2f & %.1f [%.1f, %.1f] & %.1f & %.2f & %.2f & %.0f \\\\" % (
            lab if i == 0 else "", NAME[pol], med(rs, "p50"), med(rs, "p95"), med(rs, "p99"), q1(rs, "p99"), q3(rs, "p99"),
            med(rs, "slo_miss"), med(rs, "bg_thpt"), med(rs, "bg_slow_p99"), med(rs, "bg_dly_max")))
        num[f"{wl}.{pol}"] = {k: med(rs, k) for k in ["p50", "p95", "p99", "slo_miss", "bg_thpt", "bg_slow_p99", "bg_dly_max", "trans_per_s", "ovh_pct", "util", "censored"]}
        num[f"{wl}.{pol}"]["mode_frac"] = np.median([r["mode_frac"] for r in rs], axis=0).tolist()
    rows.append("\\midrule" if wl == "mixed_burst" else "")
main_tab = "\n".join(r for r in rows if r)

def fp(h):
    if h < 1e-3:
        m, e = ('%.1e' % h).split('e'); return '$%s\\times10^{%d}$' % (m, int(e))
    return '%.3f' % h

# ---------- paired tests (Holm over 3 comparisons per workload) ----------
prow = []
for wl, lab in [("mixed_burst", "Bursty"), ("mixed_smooth", "Smooth")]:
    res = [(b, *paired(wl, b)) for b in ["eevdf", "lavd", "static"]]
    order = np.argsort([r[4] for r in res]); holm = [0] * 3; run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, res[idx][4] * (3 - rank))); holm[idx] = run
    for (b, m, lo, hi, p), h in zip(res, holm):
        prow.append("%s & vs.\\ %s & %+.1f\\%% & [%+.1f, %+.1f] & %s \\\\" % (lab, NAME[b], m, lo, hi, fp(h)))
        num[f"paired.{wl}.{b}"] = {"med_pct": m, "lo": lo, "hi": hi, "p_holm": h}
    prow.append("\\midrule" if wl == "mixed_burst" else "")
paired_tab = "\n".join(r for r in prow if r)
# ---------- single-class workloads ----------
srow = []
for wl, lab in [("A_faas_only", "A (FaaS only)"), ("B_bg_only", "B (daemons only)")]:
    for i, pol in enumerate(POL):
        rs = e2[wl][pol]
        if wl.startswith("A"):
            srow.append("%s & %s & %.2f & %.2f & %.1f & -- \\\\" % (lab if i == 0 else "", NAME[pol], med(rs, "p50"), med(rs, "p99"), med(rs, "slo_miss")))
        else:
            srow.append("%s & %s & -- & -- & -- & %.2f \\\\" % (lab if i == 0 else "", NAME[pol], med(rs, "bg_thpt")))
        num[f"{wl}.{pol}"] = {k: med(rs, k) for k in ["p50", "p99", "slo_miss", "bg_thpt", "ovh_pct"]}
    if wl.startswith("A"): srow.append("\\midrule")
single_tab = "\n".join(srow)

# ---------- ablation ----------
arow = []
for name, rs in e4.items():
    arow.append("%s & %.1f & %.1f & %.0f & %.0f \\\\" % (name.replace("%", "\\%"), med(rs, "p99"), med(rs, "slo_miss"), med(rs, "bg_dly_max"), med(rs, "trans_per_s")))
    num[f"abl.{name}"] = {k: med(rs, k) for k in ["p99", "slo_miss", "bg_dly_max", "trans_per_s", "bg_thpt"]}
abl_tab = "\n".join(arow)

# ---------- gamma sensitivity ----------
grow = []
for g, d in e5.items():
    grow.append("%s & %.1f & %.1f & %.1f \\\\" % (g, med(d["eevdf"], "p99"), med(d["static"], "p99"), med(d["ia"], "p99")))
    num[f"gamma.{g}"] = {p: med(d[p], "p99") for p in d}
gamma_tab = "\n".join(grow)

with open(os.path.join(ROOT, "paper", "tables_generated.tex"), "w") as f:
    for nm, body in [("MAINTAB", main_tab), ("PAIREDTAB", paired_tab), ("SINGLETAB", single_tab), ("ABLTAB", abl_tab), ("GAMMATAB", gamma_tab)]:
        f.write("\\newcommand{\\%s}{%%\n%s\n}\n" % (nm, body))
num["meta"] = meta; num["tuned"] = tuned
json.dump(num, open(os.path.join(R, "numbers.json"), "w"), indent=1, default=float)

# ---------- figures ----------
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
col = {"eevdf": "#7a6fb0", "static": "#d9622b", "lavd": "#888888", "ia": "#2e9e7e"}
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for pol in POL:
    lat = np.sort(np.load(os.path.join(R, f"lat_{pol}.npz"))["lat"])
    y = 1.0 - np.arange(len(lat)) / len(lat)
    ax.plot(lat, y, label=NAME[pol], color=col[pol], lw=1.2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 1); ax.set_xlim(0.15, 500)
ax.axvline(2.0, color="k", ls=":", lw=0.6)
ax.set_xlabel("FaaS latency (ms)"); ax.set_ylabel("P(latency > x)"); ax.legend(frameon=False)
ax.grid(alpha=.3, ls=":"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ccdf.pdf"))

fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.2))
mults = list(e3["burst_mult"].keys())
for pol in POL:
    axs[0].plot([float(m) for m in mults], [med(e3["burst_mult"][m][pol], "p99") for m in mults], "o-", ms=3, label=NAME[pol], color=col[pol])
    axs[1].plot([float(m) for m in mults], [med(e3["burst_mult"][m][pol], "bg_dly_max") for m in mults], "o-", ms=3, color=col[pol])
    ds = list(e3["n_bg"].keys())
    axs[2].plot([int(d) for d in ds], [med(e3["n_bg"][d][pol], "p99") for d in ds], "o-", ms=3, color=col[pol])
axs[0].set_xlabel("Burst intensity (peak/mean rate)"); axs[0].set_ylabel("FaaS p99 (ms)")
axs[1].set_xlabel("Burst intensity (peak/mean rate)"); axs[1].set_ylabel("Max bg dispatch delay (ms)")
axs[2].set_xlabel("Number of background daemons"); axs[2].set_ylabel("FaaS p99 (ms)")
for a in axs: a.grid(alpha=.3, ls=":")
axs[0].set_yscale("log"); axs[2].set_yscale("log")
axs[0].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_sweeps.pdf"))
print("ok")
