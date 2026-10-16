"""Fraction of tire grip used (|Fy| / Fy_max = |tanh(.)|) along mppi_alone episodes at a given target speed."""
import sys, os, numpy as np
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [root, os.path.join(root, "src")]
import stress_grid as sg
v, K, seeds = float(sys.argv[1]), int(sys.argv[2]), [int(s) for s in sys.argv[3].split(",")]
use = []
Env = sg.ClutteredRacingEnv
class E(Env):
    def step(self, u, dt=0.02):
        st, info = super().step(u, dt=dt)
        p = self.model.p; px, py, psi, vx, vy, r = st; L = p.lf + p.lr
        ax, d = self.actual_ax, self.actual_delta
        vxs = max(vx, p.v_min)
        af = -np.arctan2(vy + p.lf * r, vxs) + d; ar = -np.arctan2(vy - p.lr * r, vxs)
        Fzf = max(p.mass * p.g * p.lr / L - p.mass * ax * p.h / L, 1.0); Fzr = max(p.mass * p.g * p.lf / L + p.mass * ax * p.h / L, 1.0)
        rem = np.sqrt(max(p.mu ** 2 - (ax / p.g) ** 2, 0.0))
        uf = abs(np.tanh(p.mu * Fzf * p.C_Sf * af / max(Fzf * rem, 1e-3))); ur = abs(np.tanh(p.mu * Fzr * p.C_Sr * ar / max(Fzr * rem, 1e-3)))
        use.append((max(uf, ur), abs(vx * r)))
        return st, info
sg.ClutteredRacingEnv = E
outs = []
for s in seeds:
    r = sg.run_episode("mppi_alone", s, 1.0, v, K, "cuda"); outs.append(r["outcome"])
u = np.array(use)
print(f"v={v} K={K} seeds={len(seeds)} outcomes={ {o: outs.count(o) for o in set(outs)} } steps={len(u)}")
print(f"  grip used (max of front/rear): median {np.median(u[:,0]):.2f}  p90 {np.percentile(u[:,0],90):.2f}  p99 {np.percentile(u[:,0],99):.2f}")
print(f"  steps with >90% grip used: {np.mean(u[:,0]>0.9)*100:.1f}%   >97%: {np.mean(u[:,0]>0.97)*100:.1f}%")
print(f"  lateral accel |vx*r|: p90 {np.percentile(u[:,1],90):.1f}  p99 {np.percentile(u[:,1],99):.1f} m/s^2 (mu*g = 9.8)")
