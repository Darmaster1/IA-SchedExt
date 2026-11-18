import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
import numpy as np
from iasched.sim import Params, Sim, run_one, Task, FAIR, STRICT, AGING

FAST = Params(sim_ms=1500, warm_ms=300, drain_ms=1000)


def test_determinism():
    a = run_one(("ia", FAST, 3, False)); b = run_one(("ia", FAST, 3, False))
    assert a["p99"] == b["p99"] and a["bg_thpt"] == b["bg_thpt"]


def test_common_random_numbers_same_arrivals():
    s1 = Sim("eevdf", FAST, 5); s2 = Sim("static", FAST, 5)
    assert s1.faas == s2.faas


def test_latency_at_least_service_and_waits_nonnegative():
    s = Sim("eevdf", FAST, 1); m, lat = s.run()
    assert (np.array(s.waits) >= -1e-6).all()
    assert lat.min() * 1000 >= FAST.faas_min - 1e-6      # sojourn >= shortest service


def test_bg_only_is_work_conserving():
    p = replace(FAST, faas_rate=0.0, n_bg=8)
    m, _ = Sim("eevdf", p, 1).run()
    assert m["bg_thpt"] > 0.95 * p.cores * 0.95          # ~all cores busy with daemons


def test_static_priority_beats_eevdf_on_faas_tail_when_mixed():
    e = run_one(("eevdf", FAST, 2, False)); s = run_one(("static", FAST, 2, False))
    assert s["p99"] <= e["p99"]


def test_ia_mode_rules():
    p = replace(FAST, f_lo=0.3, f_hi=0.6, theta=0.05)
    s = Sim("ia", p, 1)
    def fill(nf, nb, overdue=0):
        s.ready = []
        for i in range(nf):
            t = Task(i, 0, 0.0, 100.0); t.chat = 0; t.ready_since = 0.0
            t.wait = p.slo + 1 if i < overdue else 0.0; s.ready.append(t)
        for i in range(nb):
            t = Task(-1 - i, 1, 0.0, 1e4); t.chat = 1; s.ready.append(t)
        s.win = []
    fill(1, 9); s.update_mode(10.0); assert s.mode == FAIR            # F=0.1
    fill(5, 5); s.update_mode(20.0); assert s.mode == AGING           # F=0.5
    fill(9, 1, overdue=0); s.update_mode(30.0); assert s.mode == AGING  # F=0.9 but no SLO misses
    fill(9, 1, overdue=5); s.update_mode(40.0); assert s.mode == STRICT  # F=0.9 and R_miss>theta


def test_modes_ablation_never_leaves_fair():
    p = replace(FAST, modes=(True, False, False))
    m, _ = Sim("ia", p, 1).run()
    assert m["trans_per_s"] == 0.0 and m["mode_frac"][0] == 1.0


def test_misclassification_flips_classes():
    s = Sim("ia", replace(FAST, misclass=1.0), 1)
    assert all(x[2] for x in s.faas)


def test_overhead_only_charged_for_ia():
    assert run_one(("eevdf", FAST, 1, False))["ovh_pct"] == 0.0
    assert run_one(("ia", FAST, 1, False))["ovh_pct"] > 0.0
