"""
predictive_filter.py -- Multi-step predictive safety filter (backup-trajectory / gatekeeper style).

Why: the one-step CBF-QP shield (safe_controller.py) linearizes relative-degree-2 constraints at the
current state. Past tire saturation the steering authority of those rows inverts sign, so the QP steers
INTO the wall/obstacle it protects and overrides MPPI's counter-steer recovery (spin -> stall/collision).

What: every step, roll out a small batch of candidate control sequences through the full nonlinear
dynamic bicycle model WITH the env's actuator model (steering slew limit + first-order motor lag) and
check exact 3-disk geometric clearance to obstacles and walls over the horizon. No linearization.

  candidates = first action (grid inside the 1-step reachable slew window around u_nom)
             x tail family (MPPI's own remaining plan | straighten | steer-to-center+brake | brake straight)

  (+ optionally MPPI's lowest-cost sampled rollouts as extra full-horizon candidates)

  if MPPI's own sequence is safe          -> pass u_nom through unchanged (minimal intervention)
  elif some candidate sequence is safe    -> safe candidate with the best progress-vs-deviation score
                                             (select="progress"; "closest" = first action closest to u_nom)
  else                                    -> candidate with the largest worst-case clearance

Backup (gatekeeper) mode, backup=True: each candidate is committed only for `horizon` steps and then
continued by a brake-to-stop backup (steering held, or straightened); the WHOLE sequence must be clear.
A stopped car is an invariant safe state, so a candidate accepted now leaves a known-safe continuation for
the next step (recursive feasibility up to state/perception noise and model mismatch). Without it, a fixed
0.8 s horizon can accept plans that run into dead ends just beyond the horizon.

Optional terminal condition (terminal_net): a barrier network h(x_H) >= 0 at the end of the horizon,
w.r.t. the nearest obstacle -- only meaningful for a network trained on the visited state range.
"""
import time
import numpy as np
import torch

from dynamic_bicycle_env import DynamicBicycleModel


class PredictiveSafetyFilter:
    def __init__(self, horizon=40, dt=0.02, track_half_width=2.0, r_disk=0.21,
                 obs_buffer=0.05, wall_buffer=0.05, sv_max=3.2, tau_motor=0.05,
                 n_delta=5, ax_offsets=(0.0, -1.5, -3.0, 1.5), terminal_net=None, r_margin=0.38,
                 n_samples=64, select="progress", w_dev=0.05, pref_clear=0.15,
                 backup=False, backup_steps=60, a_brake=6.0, device="cpu"):
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.model = DynamicBicycleModel(device=self.device)
        self.p = self.model.p
        self.H, self.dt = horizon, dt
        self.half_w = track_half_width
        self.r_disk = r_disk
        self.obs_buffer, self.wall_buffer = obs_buffer, wall_buffer
        self.sv_max, self.tau_motor = sv_max, tau_motor
        self.n_delta = n_delta
        self.ax_offsets = ax_offsets
        self.terminal_net = terminal_net
        self.r_margin = r_margin
        self.n_samples = n_samples
        self.select = select
        self.w_dev = w_dev
        self.pref_clear = pref_clear
        self.backup, self.backup_steps, self.a_brake = backup, backup_steps, a_brake

    # ------------------------------------------------------------------
    def _tails(self, state, u_first, plan_tail):
        """Return [F, H-1, 2] command tails (one per family) given the first action."""
        H1 = self.H - 1
        py = float(state[1])
        fams = []
        # 1. MPPI's own remaining plan (padded with its last action)
        if plan_tail is not None and len(plan_tail) > 0:
            t = np.asarray(plan_tail, dtype=np.float32)[:H1]
            if len(t) < H1:
                t = np.concatenate([t, np.repeat(t[-1:], H1 - len(t), axis=0)], axis=0)
            fams.append(t)
        # 2. Straighten: steer to 0, hold speed
        fams.append(np.tile(np.array([0.0, 0.0], dtype=np.float32), (H1, 1)))
        # 3. Steer toward the centerline with mild braking
        d_c = -0.25 * np.sign(py) if abs(py) > 0.05 else 0.0
        fams.append(np.tile(np.array([-1.0, d_c], dtype=np.float32), (H1, 1)))
        # 4. Brake straight
        fams.append(np.tile(np.array([-self.p.a_max * 0.6, 0.0], dtype=np.float32), (H1, 1)))
        return np.stack(fams, axis=0)

    def _rollout(self, state, act_u, cmds, backup_mode=None):
        """cmds [N, H, 2] commanded controls -> states [N, T, 6] through actuator model + RK4.
        With backup_mode [N] (0 = hold steering, 1 = straighten), H commanded steps are followed by
        backup_steps of braking to a stop (T = H + backup_steps); otherwise T = H."""
        N = cmds.shape[0]
        T = self.H + (self.backup_steps if backup_mode is not None else 0)
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).repeat(N, 1)
        a_ax = torch.full((N,), float(act_u[0]), device=self.device)
        a_d = torch.full((N,), float(act_u[1]), device=self.device)
        a_lim = min(self.p.a_max, self.p.mu * self.p.g)
        dmax = self.sv_max * self.dt
        alpha_m = self.dt / max(1e-4, self.tau_motor)
        out = torch.empty((N, T, 6), device=self.device)
        for t in range(T):
            if t < self.H:
                c_ax = cmds[:, t, 0].clamp(-a_lim, a_lim)
                c_d = cmds[:, t, 1].clamp(-self.p.delta_max, self.p.delta_max)
                last_d = c_d
            else:   # backup: brake to a stop (no reversing), steering held or straightened
                c_ax = torch.where(x[:, 3] > 0.3, torch.full_like(c_ax, -self.a_brake), torch.zeros_like(c_ax))
                c_d = torch.where(backup_mode == 1, torch.zeros_like(last_d), last_d)
            a_d = a_d + (c_d - a_d).clamp(-dmax, dmax)
            a_ax = (a_ax + alpha_m * (c_ax - a_ax)).clamp(-a_lim, a_lim)
            x = self.model.step_rk4_torch(x, torch.stack([a_ax, a_d], dim=1), dt=self.dt)
            out[:, t] = x
        return out

    def _clearance(self, traj, obstacles):
        """Worst-case clearance over the horizon per rollout (negative = violation). traj [N, H, 6]."""
        px, py, psi = traj[..., 0], traj[..., 1], traj[..., 2]
        c, s = torch.cos(psi), torch.sin(psi)
        lr, lf = self.p.lr, self.p.lf
        dys = torch.stack([py - lr * s, py, py + lf * s], dim=-1)            # [N, H, 3]
        wall = (self.half_w - self.wall_buffer) - (dys.abs() + self.r_disk)
        clear = wall.min(dim=-1).values.min(dim=-1).values                  # [N]
        if obstacles is not None and len(obstacles) > 0:
            dxs = torch.stack([px - lr * c, px, px + lf * c], dim=-1)
            ob = torch.as_tensor(np.asarray(obstacles, dtype=np.float32), device=self.device)
            ddx = dxs.unsqueeze(-1) - ob[:, 0]                              # [N, H, 3, M]
            ddy = dys.unsqueeze(-1) - ob[:, 1]
            d = torch.sqrt(ddx ** 2 + ddy ** 2) - (ob[:, 2] + self.r_disk + self.obs_buffer)
            clear = torch.minimum(clear, d.amin(dim=(1, 2, 3)))
        return clear

    def _terminal_ok(self, traj, obstacles):
        if self.terminal_net is None or obstacles is None or len(obstacles) == 0:
            return torch.ones(traj.shape[0], dtype=torch.bool, device=self.device)
        xT = traj[:, -1]                                                    # [N, 6]
        ob = torch.as_tensor(np.asarray(obstacles, dtype=np.float32), device=self.device)
        dx = xT[:, 0:1] - ob[:, 0]
        dy = xT[:, 1:2] - ob[:, 1]
        k = torch.argmin(dx ** 2 + dy ** 2, dim=1)                          # nearest obstacle
        idx = torch.arange(xT.shape[0], device=self.device)
        scale = 0.60 / (ob[k, 2] + self.r_margin)
        inp = torch.stack([dx[idx, k] * scale, dy[idx, k] * scale, xT[:, 2], xT[:, 3], xT[:, 4], xT[:, 5]], dim=1)
        with torch.no_grad():
            h = self.terminal_net(inp).squeeze(-1)
        return h >= 0.0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def filter_control(self, state, current_u, u_nom, obstacles, dt=0.02, plan_tail=None,
                       samples=None, sample_costs=None, **_):
        t0 = time.time()
        p = self.p
        u_nom = np.asarray(u_nom, dtype=np.float32)
        nom = np.array([np.clip(u_nom[0], -p.a_max, p.a_max), np.clip(u_nom[1], -p.delta_max, p.delta_max)],
                       dtype=np.float32)
        cur_d = float(current_u[1])
        dmax = self.sv_max * dt
        d_lo, d_hi = max(-p.delta_max, cur_d - dmax), min(p.delta_max, cur_d + dmax)
        d_nom = float(np.clip(nom[1], d_lo, d_hi))

        # First-action grid: nominal first, then slew-window steering x acceleration offsets
        firsts = [(float(nom[0]), d_nom)]
        for dd in np.linspace(d_lo, d_hi, self.n_delta):
            for oa in self.ax_offsets:
                firsts.append((float(np.clip(nom[0] + oa, -p.a_max, p.a_max)), float(dd)))
        firsts = np.array(firsts, dtype=np.float32)                         # [A, 2]

        tails = self._tails(state, nom, plan_tail)                           # [F, H-1, 2]
        A, F = len(firsts), len(tails)
        cmds = np.empty((A * F, self.H, 2), dtype=np.float32)
        cmds[:, 0] = np.repeat(firsts, F, axis=0)
        cmds[:, 1:] = np.tile(tails, (A, 1, 1))
        if samples is not None and self.n_samples > 0:
            smp = samples if isinstance(samples, np.ndarray) else samples.detach().cpu().numpy()
            if sample_costs is not None:
                sc = sample_costs if isinstance(sample_costs, np.ndarray) else sample_costs.detach().cpu().numpy()
                smp = smp[np.argsort(sc)[:self.n_samples]]
            else:
                smp = smp[:self.n_samples]
            smp = smp[:, :self.H].astype(np.float32)
            if smp.shape[1] < self.H:
                smp = np.concatenate([smp, np.repeat(smp[:, -1:], self.H - smp.shape[1], axis=1)], axis=1)
            cmds = np.concatenate([cmds, smp], axis=0)
        cmds_t = torch.as_tensor(cmds, device=self.device)

        if self.backup:   # each candidate twice: backup with steering held / straightened
            n0 = cmds_t.shape[0]
            cmds = np.concatenate([cmds, cmds], axis=0)
            cmds_t = torch.cat([cmds_t, cmds_t], dim=0)
            bmode = torch.cat([torch.zeros(n0, device=self.device), torch.ones(n0, device=self.device)])
            traj = self._rollout(state, current_u, cmds_t, backup_mode=bmode)
        else:
            traj = self._rollout(state, current_u, cmds_t)
        clear = self._clearance(traj, obstacles)
        safe = (clear > 0.0) & self._terminal_ok(traj, obstacles)
        safe_np, clear_np = safe.cpu().numpy(), clear.cpu().numpy()

        # Index 0 = nominal first action + family 0 (MPPI's own plan when provided)
        if self.backup:   # nominal is acceptable if EITHER of its backups is clear
            safe_np[0] = safe_np[0] or safe_np[len(safe_np) // 2]
        if safe_np[0] and plan_tail is not None:
            u_safe, status, idx = nom, "nominal_safe", 0
        elif safe_np.any():
            d = (cmds[:, 0, 0] - nom[0]) ** 2 / 4.0 + (cmds[:, 0, 1] - nom[1]) ** 2 / 0.01
            if self.select == "progress":   # forward progress over the horizon, mild preference for centerline
                tr = traj.cpu().numpy()
                progress = tr[:, self.H - 1, 0] - float(state[0])     # end of the committed segment
                d = -progress + 0.5 * np.abs(tr[:, self.H - 1, 1]) + self.w_dev * d
                # Progress only among candidates with a robust clearance margin (state/perception noise, mu
                # mismatch); otherwise the most-clearance safe candidate.
                robust = safe_np & (clear_np >= self.pref_clear)
                if not robust.any():
                    d = -clear_np.copy()
            d[~safe_np] = np.inf
            if self.select == "progress" and robust.any():
                d[~robust] = np.inf
            idx = int(np.argmin(d))
            u_safe, status = cmds[idx, 0], "safe_alternative"
        else:
            idx = int(np.argmax(clear_np))
            u_safe, status = cmds[idx, 0], "no_safe_max_clearance"

        u_safe = np.asarray(u_safe, dtype=np.float32)
        dev = u_safe - nom
        info = {
            "qp_active": status != "nominal_safe",
            "qp_status": status,
            "brake_override": False,
            "brake_needed": False,
            "wall_fallback": False,
            "n_obstacle_rows": int(len(obstacles)) if obstacles is not None else 0,
            "n_unreachable": 0,
            "wall_margin_min": float(clear_np[0]),
            "min_h_val": float(clear_np.max()),
            "wall_top_viol": False, "wall_bot_viol": False, "obstacle_viol": not bool(safe_np[0]),
            "n_safe": int(safe_np.sum()), "n_cand": int(len(safe_np)),
            "dev_from_nom": dev,
            "latency_ms": (time.time() - t0) * 1000.0,
        }
        return u_safe, info
