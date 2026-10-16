"""Trace one NICBF episode: per-step state, nominal target, executed command, barrier values, slack."""
import sys, os, json, math
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."));  sys.path[:0] = [root, os.path.join(root, "src"), os.path.join(root, "prototype_neural_ode", "dynamic")]
import numpy as np, torch
torch.set_num_threads(1)
import stress_grid as sg, nicbf_controller as nc
method, v, seed, kw = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), json.loads(sys.argv[4])
log = []
orig = nc.NeuralICBF.step
def step(self, state, u, obs, vt, u_nom=None, dt=0.02):
    cmd, info = orig(self, state, u, obs, vt, u_nom=u_nom, dt=dt)
    log.append((np.array(state), np.array(u), info["u_target"], cmd, info)); return cmd, info
nc.NeuralICBF.step = step
r = sg.run_episode(method, seed, 1.0, v, 50, "cuda", shield_kwargs={"nicbf": kw})
print(seed, r["outcome"], r["time_s"])
i1 = next((i for i, e in enumerate(log) if e[0][3] < 1.0), len(log) - 30)
for i in range(max(0, i1 - 60), min(len(log), i1 + 30), 6):
    s, u, ut, c, inf = log[i]
    print(f"{i:4d} x{s[0]:6.2f} y{s[1]:+.2f} psi{math.degrees(s[2]):+6.1f} vx{s[3]:4.2f} r{s[5]:+5.2f} | u {u[0]:+5.2f} {u[1]:+.2f} "
          f"target {ut[0]:+5.2f} {ut[1]:+.2f} cmd {c[0]:+5.2f} {c[1]:+.2f} | h_obs {inf['min_h_obs']:+.2f} h_e {inf['min_he']:+.2f} "
          f"slack {inf['slack_max']:.3f} nobs {inf['n_obs']}")
