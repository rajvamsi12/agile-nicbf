"""Replay one stress_grid episode and log full state + shield info per step (diagnostic only)."""
import os, sys, json, argparse
import numpy as np
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [root, os.path.join(root, "src")]
import stress_grid as sg

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="analytical_hocbf")
ap.add_argument("--seed", type=int, default=138)
ap.add_argument("--K", type=int, default=50)
ap.add_argument("--device", default="cuda")
ap.add_argument("--kw", default='{"r_margin": 0.38}')
ap.add_argument("--out", required=True)
a = ap.parse_args()

log = []
Env = sg.ClutteredRacingEnv
class TracedEnv(Env):
    def step(self, u, dt=0.02):
        s0 = np.array(self.state, dtype=float).tolist()
        st, info = super().step(u, dt=dt)
        log.append({"s": s0, "u": [float(x) for x in u], "info_last": TracedEnv._qi})
        TracedEnv._qi = None
        return st, info
TracedEnv._qi = None
sg.ClutteredRacingEnv = TracedEnv

get_shield = sg.get_shield
def traced_get_shield(*args, **kw):
    sh = get_shield(*args, **kw)
    f = sh.filter_control
    def fc(state, cu, un, obs, **k):
        u, qi = f(state, cu, un, obs, **k)
        TracedEnv._qi = {"u_nom": [float(x) for x in un], "status": qi["qp_status"],
                         "wall_margin": qi["wall_margin_min"], "top": qi["wall_top_viol"], "bot": qi["wall_bot_viol"],
                         "obs_viol": qi["obstacle_viol"], "n_obs_rows": qi["n_obstacle_rows"], "min_h": qi["min_h_val"]}
        return u, qi
    sh.filter_control = fc
    return sh
sg.get_shield = traced_get_shield

rec = sg.run_episode(a.method, a.seed, 1.0, 6.0, a.K, a.device, shield_kwargs=json.loads(a.kw))
json.dump({"rec": rec, "log": log}, open(a.out, "w"))
print(rec["outcome"], rec["time_s"])
