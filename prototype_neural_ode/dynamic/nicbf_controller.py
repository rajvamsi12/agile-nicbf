"""
Stage 4 -- Neural ICBF safe controller for the dynamic bicycle (paper eqs. 6, 9, 12, 13 with learned F, pi, h).

Augmented state z = (x, u), u = actuator state (ax, delta). Integral law  u_dot = phi(x,u) + v*,
  phi = d pi_theta/dx . F_theta(x,u) + (alpha_phi/2) (pi_theta(x) - u)                         (eq. 9)
  [or, with an external nominal such as MPPI: phi = (alpha_phi/2)(u_nom - u)]
For every learned state barrier h_i(x) (each nearby obstacle, both walls):
  h_e,i(x,u) = grad h_i(x) . F_theta(x,u) + alpha h_i(x)                                       (Ames et al. IV-C)
  ICBF row:  p_i^T v >= -gamma h_e,i - d h_e,i/dx . F_theta - p_i^T phi,  p_i = d h_e,i/du   (autograd through F_theta)
Grip (state-input ICBF, Def. 2): h_g(x,u) = (c mu g)^2 - a_y(x,u)^2, a_y = F_theta[vy_dot] + vx r.
Input constraints: actuator ranges (linear barriers) and actuator RATES (steering slew, motor-lag realisable a_dot).
QP: min ||W v||^2 + slack penalties (barrier rows soft, input rows hard), solved with quadprog (eq. 12).
"""
import time
import numpy as np
import torch
from torch.func import jacrev, vmap
from qpsolvers import solve_qp
from common import FNet, PiNet, HObsNet, HWallNet, load, TRACK_HALF, SV_MAX, TAU, DT, plant_params


class NeuralICBF:
    def __init__(self, alpha=2.0, gamma=5.0, alpha_phi=20.0, gamma_u=10.0, grip_frac=0.8, mu_design=1.0,
                 gamma_g=5.0, obs_range=5.0, w_slack=1e4, nominal="pi", a_rate_scale=100.0, h_offset=0.0, device="cpu"):
        self.dev = torch.device(device)
        self.F = load(FNet, "F_theta.pth", self.dev)
        self.pi = load(PiNet, "pi_theta.pth", self.dev) if nominal == "pi" else None
        self.ho = load(HObsNet, "h_obs.pth", self.dev)
        self.hw = load(HWallNet, "h_wall.pth", self.dev)
        for m in [mm for mm in (self.F, self.pi, self.ho, self.hw) if mm is not None]:
            for prm in m.parameters():
                prm.requires_grad_(False)
        p = plant_params(mu_design)
        self.a_lim, self.d_max = min(p.a_max, p.mu * p.g), p.delta_max
        self.g_max = grip_frac * mu_design * p.g
        self.alpha, self.gamma, self.alpha_phi, self.gamma_u, self.gamma_g = alpha, gamma, alpha_phi, gamma_u, gamma_g
        self.obs_range, self.w_slack, self.nominal = obs_range, w_slack, nominal
        # Enforce the learned barriers at a margin: h_tilde = h - h_offset (validation false-safe rate is 0 at 0.2)
        self.h_offset = h_offset
        # QP metric on v: rate scales (a_dot ~ a_rate_scale m/s^3, delta_dot ~ sv_max); smaller a_rate_scale = prefer steering
        self.W = np.diag([1.0 / a_rate_scale ** 2, 1.0 / SV_MAX ** 2])

    # ---- barrier vector h(x) for the current obstacle set (obstacles fixed while differentiating) ----
    def _h_vec(self, x, obs):
        parts = []
        if obs.shape[0] > 0:
            s = torch.stack([x[0] - obs[:, 0], x[1] - obs[:, 1], x[2].expand(len(obs)), x[3].expand(len(obs)),
                             x[4].expand(len(obs)), x[5].expand(len(obs)), obs[:, 2]], 1)
            parts.append(self.ho(s))
        top = torch.stack([TRACK_HALF - x[1], x[2], x[3], x[4], x[5]])
        bot = torch.stack([TRACK_HALF + x[1], -x[2], x[3], -x[4], -x[5]])      # mirror image of the bottom wall
        parts.append(self.hw(torch.stack([top, bot], 0)))
        return torch.cat(parts) - self.h_offset

    @torch.no_grad()
    def step(self, state, u_act, obstacles, v_target, u_nom=None, dt=DT):
        t0 = time.perf_counter()
        x = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.dev)
        u = torch.as_tensor(np.asarray(u_act, dtype=np.float32), device=self.dev)
        obs = np.asarray(obstacles, dtype=np.float32).reshape(-1, 3)
        if len(obs):
            obs = obs[np.hypot(obs[:, 0] - state[0], obs[:, 1] - state[1]) - obs[:, 2] < self.obs_range]
        ob = torch.as_tensor(obs, device=self.dev).reshape(-1, 3)

        Fxu = self.F(x, u)
        # ---- integral control law phi (eq. 9) ----
        if self.nominal == "pi" or u_nom is None:
            pi_x = self.pi(x, v_target)
            Jpi = jacrev(lambda xx: self.pi(xx, v_target))(x)                      # [2, 6]
            phi = Jpi @ Fxu + 0.5 * self.alpha_phi * (pi_x - u)
            u_target = pi_x
        else:
            u_target = torch.as_tensor(np.asarray(u_nom, dtype=np.float32), device=self.dev)
            phi = 0.5 * self.alpha_phi * (u_target - u)

        # ---- learned barrier rows ----
        h_fn = lambda xx: self._h_vec(xx, ob)
        h = h_fn(x)                                                                  # [M]
        def he_fn(xx, uu):
            return jacrev(h_fn)(xx) @ self.F(xx, uu) + self.alpha * h_fn(xx)       # [M]
        he = he_fn(x, u)
        dhe_dx, dhe_du = jacrev(he_fn, argnums=(0, 1))(x, u)                        # [M,6], [M,2]
        rows_p = dhe_du
        rows_q = -self.gamma * he - dhe_dx @ Fxu - dhe_du @ phi                     # p^T v >= q
        # ---- grip row (state-input ICBF): h_g = g_max^2 - a_y^2 ----
        def hg_fn(xx, uu):
            ay = self.F(xx, uu)[4] + xx[3] * xx[5]
            return self.g_max ** 2 - ay ** 2
        hg = hg_fn(x, u)
        dg_dx, dg_du = jacrev(hg_fn, argnums=(0, 1))(x, u)
        g_p = dg_du / self.g_max ** 2
        g_q = (-self.gamma_g * hg - dg_dx @ Fxu - dg_du @ phi) / self.g_max ** 2

        P = torch.cat([rows_p, g_p.unsqueeze(0)]).cpu().numpy().astype(np.float64)
        Q = torch.cat([rows_q, g_q.unsqueeze(0)]).cpu().numpy().astype(np.float64)
        m = len(Q)
        ph = phi.cpu().numpy().astype(np.float64); uu = u.cpu().numpy().astype(np.float64)
        # ---- input constraints as bounds on u_dot = phi + v (hard) ----
        lo = np.array([(-self.a_lim - uu[0]) / TAU, -SV_MAX], dtype=np.float64)
        hi = np.array([(self.a_lim - uu[0]) / TAU, SV_MAX], dtype=np.float64)
        # actuator-range barriers (Ames' h_u): d/dt(delta_max - delta) >= -gamma_u (delta_max - delta), etc.
        lo[1] = max(lo[1], -self.gamma_u * (uu[1] + self.d_max))
        hi[1] = min(hi[1], self.gamma_u * (self.d_max - uu[1]))
        lo[0] = max(lo[0], -self.gamma_u * (uu[0] + self.a_lim))
        hi[0] = min(hi[0], self.gamma_u * (self.a_lim - uu[0]))
        lo, hi = lo - ph, hi - ph                                                     # bounds on v
        # decision z = [v (2), slack (m)];  P_i v + s_i >= Q_i,  s_i >= 0
        Hq = np.zeros((2 + m, 2 + m)); Hq[:2, :2] = 2 * self.W; Hq[2:, 2:] = 2 * self.w_slack * np.eye(m)
        G = np.zeros((m, 2 + m)); G[:, :2] = -P; G[:, 2:] = -np.eye(m)
        lb = np.r_[lo, np.zeros(m)]; ub = np.r_[np.maximum(hi, lo), np.full(m, 1e9)]
        status = "ok"
        try:
            z = solve_qp(Hq, np.zeros(2 + m), G, -Q, lb=lb, ub=ub, solver="quadprog")
        except Exception as e:
            z, status = None, f"exception:{type(e).__name__}"
        if z is None:
            z = np.r_[np.clip(np.zeros(2), lo, np.maximum(hi, lo)), np.zeros(m)]
            status = status if status != "ok" else "infeasible"
        v = z[:2]; slack = z[2:]
        udot = ph + v
        u_next = uu + udot * dt
        cmd = np.array([np.clip(uu[0] + TAU * udot[0], -self.a_lim, self.a_lim),       # motor lag realises a_dot
                        np.clip(u_next[1], -self.d_max, self.d_max)], dtype=np.float32)
        hv = h.cpu().numpy()
        info = {"qp_status": status, "qp_active": bool(np.abs(v).max() > 1e-6),
                "min_h_obs": float(hv[:-2].min()) if len(hv) > 2 else 1.0, "min_h_wall": float(hv[-2:].min()),
                "min_he": float(he.min().item()), "h_grip": float(hg.item()), "slack_max": float(slack.max()),
                "n_obs": int(len(obs)), "v_norm": float(np.linalg.norm(v / np.array([100.0, SV_MAX]))),
                "u_target": u_target.cpu().numpy(), "latency_ms": (time.perf_counter() - t0) * 1000.0}
        return cmd, info
