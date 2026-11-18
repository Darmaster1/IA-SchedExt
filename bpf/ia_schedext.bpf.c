/* IA-SchedExt -- REFERENCE SKETCH of the three-mode policy as a sched_ext scheduler.
 *
 * STATUS: NOT COMPILED, NOT LOADED, NOT EVALUATED. No result in the paper comes from this file; all
 * reported numbers come from the discrete-event simulator in ../iasched. It is provided to show how the
 * simulated policy maps onto sched_ext hooks. Kfunc names follow Linux 6.12 (scx_bpf_dispatch /
 * scx_bpf_consume); they were renamed in later kernels (scx_bpf_dsq_insert / scx_bpf_dsq_move_to_local).
 * Build against the scx tree (scx/common.bpf.h) and verify on a 6.12+ kernel before relying on it.
 */
#include <scx/common.bpf.h>

char _license[] SEC("license") = "GPL";

#define FAAS_DSQ   0
#define BG_DSQ     1
#define SLICE_NS   2250000ULL            /* EEVDF base slice for 4 CPUs */
#define SLO_NS     2000000ULL            /* 2 ms queueing-delay SLO */

enum mode { M_FAIR, M_STRICT, M_AGING };

/* cgroup ids of FaaS runtimes, populated from user space */
struct { __uint(type, BPF_MAP_TYPE_HASH); __uint(max_entries, 1024);
         __type(key, u64); __type(value, u8); } faas_cgroups SEC(".maps");

struct task_ctx { u64 enq_ns; u8 is_faas; };
struct { __uint(type, BPF_MAP_TYPE_TASK_STORAGE); __uint(map_flags, BPF_F_NO_PREALLOC);
         __type(int, struct task_ctx); } task_ctxs SEC(".maps");

/* telemetry (a production version would use per-CPU counters) */
u64 nr_faas, nr_bg, win_done, win_miss;
u32 cur_mode = M_FAIR;
const volatile u32 f_lo_pct = 30, f_hi_pct = 60, theta_pm = 50;   /* 0.30, 0.60, 0.05 */

static __always_inline bool classify(struct task_struct *p)
{
	struct cgroup *cg = scx_bpf_task_cgroup(p);
	u64 id = cg->kn->id; u8 *hit = bpf_map_lookup_elem(&faas_cgroups, &id);
	bpf_cgroup_release(cg);
	return hit != NULL;
}

static __always_inline void update_mode(void)
{
	u64 n = nr_faas + nr_bg, f_pct = n ? 100 * nr_faas / n : 0;
	u64 miss_pm = win_done ? 1000 * win_miss / win_done : 0;
	if (f_pct > f_hi_pct && miss_pm > theta_pm)      cur_mode = M_STRICT;
	else if (f_pct < f_lo_pct)                       cur_mode = M_FAIR;
	else                                             cur_mode = M_AGING;
}

s32 BPF_STRUCT_OPS_SLEEPABLE(ia_init)
{
	scx_bpf_create_dsq(FAAS_DSQ, -1);
	return scx_bpf_create_dsq(BG_DSQ, -1);
}

void BPF_STRUCT_OPS(ia_enqueue, struct task_struct *p, u64 enq_flags)
{
	struct task_ctx *tc = bpf_task_storage_get(&task_ctxs, p, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (!tc) { scx_bpf_dispatch(p, SCX_DSQ_GLOBAL, SLICE_NS, enq_flags); return; }
	tc->is_faas = classify(p); tc->enq_ns = bpf_ktime_get_ns();
	__sync_fetch_and_add(tc->is_faas ? &nr_faas : &nr_bg, 1);
	update_mode();
	if (tc->is_faas) {
		scx_bpf_dispatch(p, FAAS_DSQ, SLICE_NS, enq_flags);
		if (cur_mode == M_STRICT)                     /* preempt a CPU running background work */
			scx_bpf_kick_cpu(scx_bpf_task_cpu(p), SCX_KICK_PREEMPT);
	} else {
		/* AGING: dispatch order is approximated by FIFO + vtime; the simulator uses
		 * V_bg - alpha*dwell, which needs a per-task vtime update at consume time. */
		scx_bpf_dispatch_vtime(p, BG_DSQ, SLICE_NS, p->scx.dsq_vtime, enq_flags);
	}
}

void BPF_STRUCT_OPS(ia_dispatch, s32 cpu, struct task_struct *prev)
{
	if (cur_mode == M_FAIR) {                          /* FAIR: serve both queues, FaaS first as tie-break */
		if (scx_bpf_consume(FAAS_DSQ) || scx_bpf_consume(BG_DSQ)) return;
	} else if (scx_bpf_consume(FAAS_DSQ)) {            /* STRICT/AGING: FaaS first */
		return;
	}
	scx_bpf_consume(BG_DSQ);
}

void BPF_STRUCT_OPS(ia_running, struct task_struct *p)
{
	struct task_ctx *tc = bpf_task_storage_get(&task_ctxs, p, 0, 0);
	if (tc && tc->is_faas) {
		__sync_fetch_and_add(&win_done, 1);
		if (bpf_ktime_get_ns() - tc->enq_ns > SLO_NS) __sync_fetch_and_add(&win_miss, 1);
	}
	/* window decay (halving every ~100 ms) is done by a user-space timer in a real build */
}

void BPF_STRUCT_OPS(ia_quiescent, struct task_struct *p, u64 deq_flags)
{
	struct task_ctx *tc = bpf_task_storage_get(&task_ctxs, p, 0, 0);
	if (tc) __sync_fetch_and_sub(tc->is_faas ? &nr_faas : &nr_bg, 1);
}

SCX_OPS_DEFINE(ia_ops, .enqueue = (void *)ia_enqueue, .dispatch = (void *)ia_dispatch,
               .running = (void *)ia_running, .quiescent = (void *)ia_quiescent,
               .init = (void *)ia_init, .name = "ia_schedext");
