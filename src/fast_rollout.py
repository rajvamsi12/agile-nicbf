"""
fast_rollout.py -- Batched NumPy twin of DynamicBicycleModel.continuous_dynamics_torch / step_rk4_torch.

The predictive filter's rollouts are overhead-bound (~0.36 ms per torch RK4 step at ANY batch size on CPU),
so latency scales with the number of simulated steps. Plain NumPy has far lower per-op overhead on small
batches. Equations are identical to dynamic_bicycle_env.py (load transfer, Kamm circle tanh saturation).
"""
import numpy as np


class FastBicycle:
    def __init__(self, p):
        self.p = p
        self.L = p.lf + p.lr

    def f(self, x, u):
        p = self.p
        psi, vx, vy, r = x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        ax, d = u[:, 0], u[:, 1]
        vxs = np.maximum(vx, p.v_min)
        af = -np.arctan2(vy + p.lf * r, vxs) + d
        ar = -np.arctan2(vy - p.lr * r, vxs)
        Fzf = np.maximum(p.mass * p.g * (p.lr / self.L) - p.mass * ax * p.h / self.L, 1.0)
        Fzr = np.maximum(p.mass * p.g * (p.lf / self.L) + p.mass * ax * p.h / self.L, 1.0)
        rem = np.sqrt(np.maximum(p.mu ** 2 - (ax / p.g) ** 2, 0.0))
        mf, mr = np.maximum(Fzf * rem, 1e-3), np.maximum(Fzr * rem, 1e-3)
        Fyf = Fzf * rem * np.tanh(p.mu * Fzf * p.C_Sf * af / mf)
        Fyr = Fzr * rem * np.tanh(p.mu * Fzr * p.C_Sr * ar / mr)
        c, s, cd, sd = np.cos(psi), np.sin(psi), np.cos(d), np.sin(d)
        out = np.empty_like(x)
        out[:, 0] = vx * c - vy * s
        out[:, 1] = vx * s + vy * c
        out[:, 2] = r
        out[:, 3] = ax + vy * r - Fyf * sd / p.mass
        out[:, 4] = -vx * r + (Fyf * cd + Fyr) / p.mass
        out[:, 5] = (p.lf * Fyf * cd - p.lr * Fyr) / p.Iz
        return out

    def step_rk4(self, x, u, dt):
        """Mirrors step_rk4_torch: control clamping, vx >= 0, vy = r = 0 at standstill."""
        p = self.p
        a_lim = min(p.a_max, p.mu * p.g)
        u = np.column_stack([np.clip(u[:, 0], -a_lim, a_lim), np.clip(u[:, 1], -p.delta_max, p.delta_max)])
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * dt * k1, u)
        k3 = self.f(x + 0.5 * dt * k2, u)
        k4 = self.f(x + dt * k3, u)
        n = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        n[:, 3] = np.maximum(n[:, 3], 0.0)
        z = n[:, 3] <= 1e-4
        n[z, 4] = 0.0
        n[z, 5] = 0.0
        return n
