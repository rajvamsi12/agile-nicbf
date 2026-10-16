"""Trace predictive_filter decisions per step for one episode (diagnostic only)."""
import sys, os, math, json, argparse, numpy as np
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [root, os.path.join(root, "src")]
import stress_grid as sg, predictive_filter as pfm
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int); ap.add_argument("--K", type=int, default=1000)
ap.add_argument("--psf", default='{"backup": true, "horizon": 20, "carry": true}'); ap.add_argument("--out")
a = ap.parse_args()
log = []
orig = pfm.PredictiveSafetyFilter.filter_control
def fc(self, state, cu, un, obs, **k):
    u, qi = orig(self, state, cu, un, obs, **k)
    log.append({"s": [float(x) for x in state], "nom": [float(x) for x in un], "u": [float(x) for x in u],
                "status": qi["qp_status"], "nom_clear": qi["wall_margin_min"], "best_clear": qi["min_h_val"], "n_safe": qi["n_safe"], "n_cert_fail": qi.get("n_cert_fail", 0), "obs": [[float(v) for v in o] for o in obs]})
    return u, qi
pfm.PredictiveSafetyFilter.filter_control = fc
rec = sg.run_episode("predictive_filter", a.seed, 1.0, 6.0, a.K, "cuda", shield_kwargs={"psf": json.loads(a.psf)})
json.dump({"rec": rec, "log": log}, open(a.out, "w")); print(a.seed, rec["outcome"], rec["time_s"])
