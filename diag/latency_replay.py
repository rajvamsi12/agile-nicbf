"""Isolated filter latency: record real filter inputs from one episode, then replay them through filter configs.

  record: python diag/latency_replay.py record <seed> <K> <out.pt>
  replay: python diag/latency_replay.py replay <in.pt>      (run on an idle machine; uses OMP_NUM_THREADS)
"""
import sys, os, json, time, numpy as np
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [root, os.path.join(root, "src")]
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
import predictive_filter as pfm

BASE = {"backup": True, "horizon": 20, "carry": True, "max_heading": 1.4, "backup_yaw_gain": 0.1, "a_brake": 4.5}
NET = os.path.join(root, "models", "terminal_barrier_8d.pth")
CONFIGS = {
    "v6 exact full-stop": dict(BASE),
    "v7 short+neural": dict(BASE, backup_steps=15, terminal_net_path=NET, terminal_thresh=0.2),
    "v8 screen+verify": dict(BASE, backup_steps=15, terminal_net_path=NET, terminal_thresh=0.2, verify_steps=60, verify_k=8),
    "v9 +margin": dict(BASE, backup_steps=15, terminal_net_path=NET, terminal_thresh=0.2, verify_steps=60, verify_k=8, verify_margin=0.1),
    "v6 numpy": dict(BASE, backend="numpy"),
    "v9 numpy": dict(BASE, backup_steps=15, terminal_net_path=NET, terminal_thresh=0.2, verify_steps=60, verify_k=8, verify_margin=0.1, backend="numpy"),
    "v6 numpy bdt.04": dict(BASE, backend="numpy", backup_dt=0.04),
    "v9 numpy bdt.04": dict(BASE, backup_steps=15, terminal_net_path=NET, terminal_thresh=0.2, verify_steps=60, verify_k=8, verify_margin=0.1, backend="numpy", backup_dt=0.04),
}
CONFIGS["v10 neural screen"] = json.load(open(os.path.join(root, "diag", "cfg_v10.json")))
CONFIGS["v10x exact"] = json.load(open(os.path.join(root, "diag", "cfg_v10x.json")))
if len(sys.argv) > 3:
    CONFIGS = {k: v for k, v in CONFIGS.items() if any(t in k for t in sys.argv[3].split(","))}

if sys.argv[1] == "record":
    import stress_grid as sg
    seed, K, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rec = []
    orig = pfm.PredictiveSafetyFilter.filter_control
    def fc(self, state, cu, un, obs, dt=0.02, plan_tail=None, samples=None, sample_costs=None, **k):
        sc = sample_costs.detach().cpu().numpy() if sample_costs is not None else None
        sm = samples.detach().cpu().numpy() if samples is not None else None
        top = np.argsort(sc)[:64] if sc is not None else slice(0, 64)
        rec.append(dict(state=np.array(state), cu=np.array(cu), un=np.array(un), obs=np.array(obs),
                        tail=np.array(plan_tail), smp=sm[top] if sm is not None else None,
                        sc=sc[top] if sc is not None else None))
        return orig(self, state, cu, un, obs, dt=dt, plan_tail=plan_tail, samples=samples, sample_costs=sample_costs)
    pfm.PredictiveSafetyFilter.filter_control = fc
    r = sg.run_episode("predictive_filter", seed, 1.0, 6.0, K, "cuda", shield_kwargs={"psf": CONFIGS["v6 exact full-stop"]})
    torch.save(rec, out)
    print("recorded", len(rec), "steps;", r["outcome"])
else:
    rec = torch.load(sys.argv[2], weights_only=False)
    for name, cfg in CONFIGS.items():
        f = pfm.PredictiveSafetyFilter(device="cpu", **cfg)
        for e in rec[:5]:   # warm-up
            f.filter_control(e["state"], e["cu"], e["un"], e["obs"], plan_tail=e["tail"], samples=e["smp"], sample_costs=e["sc"])
        f = pfm.PredictiveSafetyFilter(device="cpu", **cfg)
        lat = []
        for e in rec:
            t = time.perf_counter()
            f.filter_control(e["state"], e["cu"], e["un"], e["obs"], plan_tail=e["tail"], samples=e["smp"], sample_costs=e["sc"])
            lat.append((time.perf_counter() - t) * 1000)
        lat = np.array(lat)
        print(f"{name:20s} threads={torch.get_num_threads()} mean {lat.mean():5.1f} ms  p50 {np.median(lat):5.1f}  "
              f"p95 {np.percentile(lat, 95):5.1f}  max {lat.max():6.1f}  | >20 ms on {np.mean(lat > 20) * 100:4.1f}% of steps")
