"""Run one episode with predictive_filter and report outcome, status mix and filter latency."""
import sys, os, json, collections, numpy as np
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [root, os.path.join(root, "src")]
import torch; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
import stress_grid as sg, predictive_filter as pfm
seed, K, psf = int(sys.argv[1]), int(sys.argv[2]), json.loads(sys.argv[3])
st, lat = collections.Counter(), []
orig = pfm.PredictiveSafetyFilter.filter_control
def fc(self, *a, **k):
    u, qi = orig(self, *a, **k); st[qi["qp_status"].split(":")[0]] += 1; lat.append(qi["latency_ms"]); return u, qi
pfm.PredictiveSafetyFilter.filter_control = fc
r = sg.run_episode("predictive_filter", seed, 1.0, 6.0, K, "cuda", shield_kwargs={"psf": psf})
n = sum(st.values())
print(f"seed {seed} K={K}: {r['outcome']:<13} {r['time_s']:5.2f}s | lat mean {np.mean(lat):5.1f} p95 {np.percentile(lat,95):5.1f} ms | "
      + " ".join(f"{k}={v/n*100:.0f}%" for k, v in st.most_common()))
