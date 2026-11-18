"""Discrete-event simulator for mixed FaaS + background-daemon scheduling.

Time unit: microseconds. Policies: 'eevdf', 'static', 'lavd', 'ia'.
All modelling assumptions are collected in `Params` and documented in README.md.
"""
import heapq
from dataclasses import dataclass, replace
import numpy as np

INF = 1e18
FAIR, STRICT, AGING = 0, 1, 2
END, BGWAKE = 0, 1


@dataclass
class Params:
    # machine
    cores: int = 4
    q_base: float = 2250.0      # EEVDF base slice: 0.75 ms * (1 + ilog2(4 cpus))
    cs: float = 3.0             # context-switch cost (us), charged when core switches task
    cold: float = 40.0          # cold-cache refill when FaaS follows a bg task on a core (us)
    gamma: float = 0.30         # FaaS service inflation when all other cores run bg (LLC/mem-BW)
    slo: float = 2000.0         # SLO: total queueing delay of an invocation <= 2 ms
    # workload
    faas_rate: float = 1500.0   # invocations / s (0 disables FaaS)
    faas_median: float = 800.0  # us, log-normal median
    faas_sigma: float = 1.0
    faas_min: float = 150.0
    faas_max: float = 12000.0
    burst_mult: float = 1.0     # >1 enables bursty (modulated) arrivals, mean rate preserved
    burst_period: float = 1000000.0
    burst_duty: float = 0.2
    n_bg: int = 4               # persistent CPU-bound daemons (0 disables)
    bg_work: tuple = (10000.0, 50000.0)
    bg_sleep: tuple = (500.0, 2000.0)
    sim_ms: float = 4000.0
    warm_ms: float = 500.0
    drain_ms: float = 1500.0
    # IA-SchedExt
    f_lo: float = 0.30
    f_hi: float = 0.60
    theta: float = 0.05
    win: float = 100000.0       # SLO-miss telemetry window (us)
    boost: float = 4.0          # AGING: FaaS deadline = q/boost ahead of V
    alpha: float = 0.5          # AGING: V_bg_aged = V_bg - alpha * dwell
    t_gran: float = 1000.0      # AGING: min run before a bg task may be preempted
    c_tel: float = 0.2          # modelled per-enqueue telemetry cost (us)  [ASSUMED]
    c_sw: float = 2.0           # modelled per-mode-transition cost (us)    [ASSUMED]
    dwell_min: float = 0.0      # optional hysteresis: minimum time between mode changes (us)
    misclass: float = 0.0       # probability a task is classified into the wrong class
    modes: tuple = (True, True, True)  # enable (FAIR, STRICT, AGING) -- ablations
    n_workers: int = 64         # FaaS worker-thread pool (LAVD history keys)


class Task:
    __slots__ = ("id", "cls", "chat", "arr", "work", "rem", "vr", "dl", "q_rem",
                 "ready_since", "wait", "seg_begin", "seg_exec", "seg_len", "f",
                 "worker", "meas", "bgid")

    def __init__(self, tid, cls, arr, work):
        self.id = tid; self.cls = cls; self.chat = cls; self.arr = arr
        self.work = work; self.rem = work
        self.vr = 0.0; self.dl = 0.0; self.q_rem = 0.0
        self.ready_since = arr; self.wait = 0.0
        self.seg_begin = 0.0; self.seg_exec = 0.0; self.seg_len = 0.0; self.f = 1.0
        self.worker = 0; self.meas = False; self.bgid = -1


class Sim:
    def __init__(self, policy, p: Params, seed: int):
        self.policy = policy; self.p = p; self.seed = seed
        C = p.cores
        self.run_t = [None] * C
        self.ver = [0] * C
        self.last = [None] * C
        self.lastcls = [-1] * C
        self.ready = []
        self.heap = []
        self.seq = 0
        self.lastV = 0.0
        self.t_sim = p.sim_ms * 1000.0
        self.t_warm = p.warm_ms * 1000.0
        self.t_end = (p.sim_ms + p.drain_ms) * 1000.0
        # statistics
        self.busy = 0.0
        self.bg_work_done = 0.0
        self.lat = []; self.waits = []; self.bg_waits = []
        self.n_done = 0; self.n_meas = 0; self.n_mdone = 0; self.bg_slow = []; self.bg_dly = []
        self.ovh_total = 0.0
        # IA state
        self.mode = FAIR; self.mode_t0 = 0.0
        self.mode_time = [0.0, 0.0, 0.0]; self.transitions = 0
        self.win = []; self.pend = 0.0
        # LAVD state
        self.ewma = [2000.0] * (p.n_workers + max(p.n_bg, 1))
        self._gen()

    # ---------------- workload ----------------
    def _gen(self):
        p = self.p
        rng = np.random.default_rng(self.seed)
        arr = []
        if p.faas_rate > 0:
            hi = p.faas_rate * max(p.burst_mult, 1.0)
            n_est = int(hi * p.sim_ms / 1000.0 * 1.3) + 50
            gaps = rng.exponential(1e6 / hi, n_est)
            times = np.cumsum(gaps)
            times = times[times < self.t_sim]
            if p.burst_mult > 1.0:
                low = (1.0 - p.burst_duty * p.burst_mult) / (1.0 - p.burst_duty) * p.faas_rate
                phase = (times % p.burst_period) / p.burst_period
                keep_p = np.where(phase < p.burst_duty, 1.0, low / hi)
                times = times[rng.random(len(times)) < keep_p]
            work = rng.lognormal(np.log(p.faas_median), p.faas_sigma, len(times))
            work = np.clip(work, p.faas_min, p.faas_max)
            flips = np.random.default_rng(self.seed + 7919).random(len(times)) < p.misclass
            arr = list(zip(times.tolist(), work.tolist(), flips.tolist()))
        self.faas = arr
        self.n_meas = sum(1 for a in arr if a[0] >= self.t_warm)
        self.bg_rng = [np.random.default_rng(self.seed * 1009 + d + 1) for d in range(p.n_bg)]
        self.bg_tasks = []
        for d in range(p.n_bg):
            t = Task(-(d + 1), 1, 0.0, 0.0); t.bgid = d
            t.worker = p.n_workers + d
            if self.p.misclass > 0 and np.random.default_rng(self.seed + 31 * d).random() < p.misclass:
                t.chat = 0
            self.bg_tasks.append(t)
            # stagger first wake slightly so daemons are not phase-locked
            self._push(float(self.bg_rng[d].uniform(0, 500.0)), BGWAKE, d, 0)

    def _push(self, when, kind, a, b):
        self.seq += 1
        heapq.heappush(self.heap, (when, self.seq, kind, a, b))

    # ---------------- helpers ----------------
    def vnow(self, now):
        tot = 0.0; n = 0
        for t in self.ready:
            tot += t.vr; n += 1
        for t in self.run_t:
            if t is not None:
                e = now - t.seg_exec
                e = 0.0 if e < 0 else (t.seg_len if e > t.seg_len else e)
                tot += t.vr + e; n += 1
        if n:
            self.lastV = tot / n
        return self.lastV

    def crit(self, t):
        e = self.ewma[t.worker]
        c = 2000.0 / e
        return 0.25 if c < 0.25 else (8.0 if c > 8.0 else c)

    def slice0(self, t):
        if self.policy == "static" and t.cls == 0:
            return INF
        return self.p.q_base

    def _set_mode(self, m, now):
        if m != self.mode and (now - self.mode_t0) >= self.p.dwell_min:
            self.mode_time[self.mode] += now - self.mode_t0
            self.mode_t0 = now
            self.mode = m
            self.transitions += 1
            self.pend += self.p.c_sw

    def update_mode(self, now):
        p = self.p
        nf = nb = 0
        qf = qover = 0
        for t in self.ready:
            if t.chat == 0:
                nf += 1; qf += 1
                if t.wait + (now - t.ready_since) > p.slo:
                    qover += 1
            else:
                nb += 1
        for t in self.run_t:
            if t is not None:
                if t.chat == 0: nf += 1
                else: nb += 1
        w = self.win
        cut = now - p.win
        k = 0
        while k < len(w) and w[k][0] < cut:
            k += 1
        if k:
            del w[:k]
        viol = sum(1 for x in w if x[1]) + qover
        den = len(w) + qf
        r = viol / den if den else 0.0
        f = nf / (nf + nb) if (nf + nb) else 0.0
        en = p.modes
        if f > p.f_hi and r > p.theta and en[STRICT]:
            m = STRICT
        elif f < p.f_lo and en[FAIR]:
            m = FAIR
        else:
            m = AGING if en[AGING] else (STRICT if (en[STRICT] and f > p.f_lo) else FAIR)
        self._set_mode(m, now)

    # ---------------- policy: pick ----------------
    def _eevdf(self, cands, now):
        V = self.vnow(now)
        best = None; bd = INF
        for t in cands:
            if t.vr <= V + 1e-6 and t.dl < bd:
                best = t; bd = t.dl
        if best is None:
            for t in cands:
                if t.dl < bd:
                    best = t; bd = t.dl
        return best

    def pick(self, now):
        rd = self.ready
        if not rd:
            return None
        pol = self.policy
        if pol in ("eevdf", "lavd"):
            t = self._eevdf(rd, now)
        elif pol == "static":
            fa = [x for x in rd if x.cls == 0]
            if fa:
                t = min(fa, key=lambda x: x.ready_since)
            else:
                t = self._eevdf(rd, now)
        else:  # ia
            self.update_mode(now)
            m = self.mode
            if m == FAIR:
                t = self._eevdf(rd, now)
            elif m == STRICT:
                fa = [x for x in rd if x.chat == 0]
                t = min(fa, key=lambda x: x.ready_since) if fa else self._eevdf(rd, now)
            else:
                p = self.p; q = p.q_base
                best = None; bk = INF
                for x in rd:
                    if x.chat == 0:
                        k = x.vr + q / p.boost
                    else:
                        k = x.vr - p.alpha * (now - x.ready_since) + q
                    if k < bk:
                        bk = k; best = x
                t = best
        rd.remove(t)
        return t

    # ---------------- core control ----------------
    def start(self, c, t, now):
        p = self.p
        o = p.cs if self.last[c] is not t else 0.0
        if self.policy == "ia":
            o += self.pend
            self.ovh_total += self.pend
            self.pend = 0.0
            if self.mode == STRICT and t.chat == 0:
                t.q_rem = INF
        cold = p.cold if (t.cls == 0 and self.lastcls[c] == 1 and self.last[c] is not t) else 0.0
        f = 1.0
        if t.cls == 0:
            nb = 0
            for j in range(p.cores):
                if j != c and self.run_t[j] is not None and self.run_t[j].cls == 1:
                    nb += 1
            f = 1.0 + p.gamma * nb / (p.cores - 1)
        t.f = f
        if t.cls == 1 and self.t_warm <= now <= self.t_sim:
            self.bg_dly.append(now - t.ready_since)
        t.wait += now - t.ready_since
        t.seg_begin = now
        t.seg_exec = now + o + cold
        t.seg_len = min(t.rem * f, t.q_rem)
        self.run_t[c] = t
        self.ver[c] += 1
        self._push(t.seg_exec + t.seg_len, END, c, self.ver[c])

    def _account(self, t, run, now):
        prog = run / t.f
        t.rem -= prog; t.vr += run; t.q_rem -= run
        self.busy += now - t.seg_begin
        if t.cls == 1 and self.t_warm <= now <= self.t_sim:
            self.bg_work_done += prog

    def dispatch_idle(self, now):
        for c in range(self.p.cores):
            if self.run_t[c] is None and self.ready:
                t = self.pick(now)
                if t is None:
                    break
                self.start(c, t, now)

    def stop_core(self, c, now):
        t = self.run_t[c]
        e = now - t.seg_exec
        e = 0.0 if e < 0 else (t.seg_len if e > t.seg_len else e)
        self._account(t, e, now)
        self.ver[c] += 1
        self.run_t[c] = None
        self.last[c] = t; self.lastcls[c] = t.cls
        t.ready_since = now
        self.ready.append(t)

    def preempt_for(self, t, now):
        pol = self.policy; p = self.p
        victim = None
        if pol == "static":
            if t.cls == 0:
                for c in range(p.cores):
                    x = self.run_t[c]
                    if x is not None and x.cls == 1:
                        victim = c; break
        elif pol == "lavd":
            ct = self.crit(t); bc = None; bv = INF
            for c in range(p.cores):
                x = self.run_t[c]
                if x is not None:
                    cx = self.crit(x)
                    if cx < bv:
                        bv = cx; bc = c
            if bc is not None and ct > bv * 1.5:
                victim = bc
        elif pol == "ia":
            if t.chat == 0 and self.mode != FAIR:
                bestrun = -1.0
                for c in range(p.cores):
                    x = self.run_t[c]
                    if x is not None and x.chat == 1:
                        ran = now - x.seg_exec
                        if self.mode == STRICT or ran >= p.t_gran:
                            if ran > bestrun:
                                bestrun = ran; victim = c
        if victim is not None:
            self.stop_core(victim, now)
            self.dispatch_idle(now)

    # ---------------- events ----------------
    def wake(self, t, now):
        V = self.vnow(now)
        t.vr = V
        t.ready_since = now
        t.q_rem = self.slice0(t)
        if self.policy == "lavd":
            t.dl = V + self.p.q_base / self.crit(t)
        else:
            t.dl = V + self.p.q_base
        if self.policy == "ia":
            self.pend += self.p.c_tel
        self.ready.append(t)
        self.dispatch_idle(now)
        if t in self.ready:
            self.preempt_for(t, now)

    def on_end(self, c, ver, now):
        if ver != self.ver[c]:
            return
        p = self.p
        t = self.run_t[c]
        self._account(t, t.seg_len, now)
        self.run_t[c] = None
        self.last[c] = t; self.lastcls[c] = t.cls
        if t.rem <= 1e-6:
            if t.cls == 0:
                self.n_done += 1
                if t.meas:
                    self.n_mdone += 1
                    self.lat.append(now - t.arr)
                    self.waits.append(t.wait)
                self.win.append((now, t.wait > p.slo))
                e = self.ewma[t.worker]
                self.ewma[t.worker] = 0.75 * e + 0.25 * t.work
            else:
                e = self.ewma[t.worker]
                self.ewma[t.worker] = 0.75 * e + 0.25 * t.work
                if self.t_warm <= now <= self.t_sim:
                    self.bg_slow.append((now - t.arr) / t.work)
                rng = self.bg_rng[t.bgid]
                gap = float(rng.uniform(*p.bg_sleep))
                self._push(now + gap, BGWAKE, t.bgid, 0)
        else:
            # slice expired: renew deadline, requeue
            t.q_rem = p.q_base
            if self.policy == "lavd":
                t.dl = t.vr + p.q_base / self.crit(t)
            else:
                t.dl = t.vr + p.q_base
            t.ready_since = now
            self.ready.append(t)
        self.dispatch_idle(now)

    def on_bgwake(self, d, now):
        t = self.bg_tasks[d]
        rng = self.bg_rng[d]
        t.work = float(rng.uniform(*self.p.bg_work)); t.rem = t.work
        t.wait = 0.0; t.arr = now
        self.wake(t, now)

    def run(self):
        p = self.p
        fa = self.faas; i = 0; nf = len(fa); tid = 0
        heap = self.heap
        while True:
            tf = fa[i][0] if i < nf else INF
            th = heap[0][0] if heap else INF
            if tf <= th:
                if tf >= INF:
                    break
                now = tf
                a, w, flip = fa[i]; i += 1
                t = Task(tid, 0, a, w); tid += 1
                t.worker = t.id % p.n_workers
                t.meas = a >= self.t_warm
                if flip:
                    t.chat = 1
                self.wake(t, now)
            else:
                now, _, kind, a, b = heapq.heappop(heap)
                if kind == END:
                    self.on_end(a, b, now)
                else:
                    self.on_bgwake(a, now)
            if now > self.t_end:
                break
            if i >= nf and now > self.t_sim and self.n_mdone >= self.n_meas and self.n_meas > 0:
                break
            if nf == 0 and now > self.t_sim:
                break
        self.now = now
        self.mode_time[self.mode] += max(0.0, now - self.mode_t0)
        return self.metrics()

    # ---------------- metrics ----------------
    def metrics(self):
        p = self.p
        lat = np.array(self.lat) / 1000.0
        wt = np.array(self.waits)
        out = {"policy": self.policy, "seed": self.seed}
        if len(lat):
            q = np.percentile(lat, [50, 95, 99, 99.9])
            out.update(p50=q[0], p95=q[1], p99=q[2], p999=q[3], mean=float(lat.mean()))
            out["slo_miss"] = float((wt > p.slo).mean() * 100.0)
            out["wait_p99"] = float(np.percentile(wt, 99) / 1000.0)
            # censored invocations (never finished within drain) count as SLO misses
            cens = self.n_meas - len(lat)
            out["censored"] = cens
        else:
            out.update(p50=0, p95=0, p99=0, p999=0, mean=0, slo_miss=0, wait_p99=0, censored=0)
        bs = np.array(self.bg_slow) if self.bg_slow else np.array([1.0])
        bd = np.array(self.bg_dly) / 1000.0 if self.bg_dly else np.array([0.0])
        out["bg_slow_med"] = float(np.median(bs)); out["bg_slow_p99"] = float(np.percentile(bs, 99))
        out["bg_dly_p99"] = float(np.percentile(bd, 99)); out["bg_dly_max"] = float(bd.max())
        out["bg_thpt"] = self.bg_work_done / (self.t_sim - self.t_warm)   # core-equivalents
        out["util"] = self.busy / (p.cores * self.now) if self.now else 0.0
        tot = max(sum(self.mode_time), 1e-9)
        out["mode_frac"] = [x / tot for x in self.mode_time]
        out["trans_per_s"] = self.transitions / (self.now / 1e6) if self.now else 0.0
        out["ovh_pct"] = 100.0 * self.ovh_total / (p.cores * self.now) if self.now else 0.0
        out["n_meas"] = self.n_meas
        return out, lat


def run_one(args):
    policy, params, seed, keep = args
    sim = Sim(policy, params, seed)
    m, lat = sim.run()
    if keep:
        m["lat"] = lat.tolist()
    return m
