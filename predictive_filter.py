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
The previously committed sequence, shifted by one step, is always re-checked as a candidate (carry=True):
that is what makes the recursive-feasibility argument hold in practice.

Heading constraint (max_heading): a candidate whose heading leaves +/-max_heading anywhere along the
trajectory (backup included) is rejected -- spinning to a stop is geometrically clear but ends the run
facing the wrong way (observed: hard-brake alternatives at 6 m/s -> oversteer -> U-turn). If the car is
already beyond the limit, the bound relaxes to "no worse than now".

Neural terminal certificate (terminal_net_path): with a SHORT backup (backup_steps), the backup no longer
reaches a stop; instead its end state must be certified by the learned terminal barrier
(train_terminal_barrier.py: h > 0 <=> a stopping backup maneuver exists under the same actuator model)
against every perceived obstacle, plus an analytic wall condition (lateral braking distance < clearance).

Screen-then-verify (verify_steps, option B): the neural-certified short backup is only used to RANK
candidates cheaply; the top verify_k are then re-checked with the exact full-length brake-to-stop backup
(verify_steps), and the first one that passes is executed and committed. If none passes, the filter falls
back to a full exact evaluation of every candidate (= v6 behaviour). Safety therefore rests on exact
rollouts, as in v6; the network buys latency.
Robust commitment (verify_margin): a NEW candidate is committed only if its exact clearance is >= verify_margin;
the previously committed plan (shifted) is always re-verified and stays acceptable at clearance > 0. With a
zero margin, a plan verified at +1 cm loses feasibility to state noise / friction mismatch one step later.

Legacy optional terminal condition (terminal_net): a barrier network h(x_H) >= 0 at the end of the horizon,
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
                 backup=False, backup_steps=60, a_brake=6.0, carry=True, mu_scale=1.0,
                 max_heading=None, backup_yaw_gain=None, terminal_net_path=None, terminal_thresh=0.1,
                 wall_a_lat=6.0, verify_steps=None, verify_k=8, verify_margin=0.0, backend="torch",
                 backup_dt=None, no_safe_fallback="max_clearance", device="cpu"):
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
        self.carry = carry
        self.max_heading = max_heading
        self.backup_yaw_gain = backup_yaw_gain   # if set, backup mode 1 = yaw-damping brake: delta = -k * r
        self.terminal_thresh, self.wall_a_lat = terminal_thresh, wall_a_lat
        self.cert = None
        if terminal_net_path is not None:
            from train_terminal_barrier import TerminalBarrierNet
            self.cert = TerminalBarrierNet().to(self.device)
            self.cert.load_state_dict(torch.load(terminal_net_path, map_location=self.device))
            self.cert.eval()
        self._last_ad = None
        self.verify_steps, self.verify_k, self.verify_margin = verify_steps, verify_k, verify_margin
        # backend="numpy": same dynamics (fast_rollout.FastBicycle, equal to the torch model to 1e-10) with far
        # lower per-step overhead. backup_dt: coarser integration step for the brake-to-stop backup phase.
        self.backend, self.backup_dt = backend, backup_dt
        self.no_safe_fallback = no_safe_fallback
        if backend == "numpy":
            from fast_rollout import FastBicycle
            self.fast = FastBicycle(self.p)
        self._ctx = None
        self.model.p.mu = self.p.mu * mu_scale      # < 1: assume less grip than the nominal model
        self._prev = None                           # (commanded sequence [H, 2], backup mode) last committed

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
        if self.backend == "numpy":
            return self._rollout_np(state, act_u, cmds, backup_mode)
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
                straight = (-self.backup_yaw_gain * x[:, 5]).clamp(-self.p.delta_max, self.p.delta_max) \
                    if self.backup_yaw_gain is not None else torch.zeros_like(last_d)
                c_d = torch.where(backup_mode == 1, straight, last_d)
            a_d = a_d + (c_d - a_d).clamp(-dmax, dmax)
            a_ax = (a_ax + alpha_m * (c_ax - a_ax)).clamp(-a_lim, a_lim)
            x = self.model.step_rk4_torch(x, torch.stack([a_ax, a_d], dim=1), dt=self.dt)
            out[:, t] = x
        self._last_ad = a_d
        return out

    def _rollout_np(self, state, act_u, cmds, backup_mode=None):
        """NumPy twin of _rollout (float64). The backup phase may use a coarser step backup_dt."""
        p = self.p
        cm = cmds.cpu().numpy() if torch.is_tensor(cmds) else np.asarray(cmds)
        bmode = backup_mode.cpu().numpy() if torch.is_tensor(backup_mode) else backup_mode
        N = cm.shape[0]
        n_b = 0
        if bmode is not None:
            bdt = self.backup_dt or self.dt
            n_b = int(np.ceil(self.backup_steps * self.dt / bdt - 1e-9))
        T = self.H + n_b
        x = np.tile(np.asarray(state, dtype=np.float64), (N, 1))
        a_ax = np.full(N, float(act_u[0]))
        a_d = np.full(N, float(act_u[1]))
        a_lim = min(p.a_max, p.mu * p.g)
        out = np.empty((N, T, 6))
        last_d = None
        for t in range(T):
            dt = self.dt if t < self.H else (self.backup_dt or self.dt)
            if t < self.H:
                c_ax = np.clip(cm[:, t, 0], -a_lim, a_lim)
                c_d = np.clip(cm[:, t, 1], -p.delta_max, p.delta_max)
                last_d = c_d
            else:
                c_ax = np.where(x[:, 3] > 0.3, -self.a_brake, 0.0)
                straight = np.clip(-self.backup_yaw_gain * x[:, 5], -p.delta_max, p.delta_max) \
                    if self.backup_yaw_gain is not None else np.zeros(N)
                c_d = np.where(bmode == 1, straight, last_d)
            dmax = self.sv_max * dt
            a_d = a_d + np.clip(c_d - a_d, -dmax, dmax)
            a_ax = np.clip(a_ax + dt / max(1e-4, self.tau_motor) * (c_ax - a_ax), -a_lim, a_lim)
            x = self.fast.step_rk4(x, np.column_stack([a_ax, a_d]), dt)
            out[:, t] = x
        self._last_ad = torch.as_tensor(a_d, dtype=torch.float32, device=self.device)
        return torch.as_tensor(out, dtype=torch.float32, device=self.device)

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

    def _certified(self, traj, obstacles):
        """Neural obstacle certificate + analytic wall condition at the end of each rollout."""
        xT = traj[:, -1]
        N = xT.shape[0]
        ok = torch.ones(N, dtype=torch.bool, device=self.device)
        hmin = torch.full((N,), 1.0, device=self.device)
        if obstacles is not None and len(obstacles) > 0:
            ob = torch.as_tensor(np.asarray(obstacles, dtype=np.float32), device=self.device)   # [M, 3]
            M = ob.shape[0]
            inp = torch.stack([
                (xT[:, 0:1] - ob[:, 0]).reshape(-1), (xT[:, 1:2] - ob[:, 1]).reshape(-1),
                xT[:, 2:3].expand(N, M).reshape(-1), xT[:, 3:4].expand(N, M).reshape(-1),
                xT[:, 4:5].expand(N, M).reshape(-1), xT[:, 5:6].expand(N, M).reshape(-1),
                self._last_ad.unsqueeze(1).expand(N, M).reshape(-1), ob[:, 2].expand(N, M).reshape(-1)], dim=1)
            hmin = self.cert(inp).reshape(N, M).min(dim=1).values
            ok = hmin >= self.terminal_thresh
        # Walls: lateral velocity toward the nearer wall must be brakeable within the remaining clearance
        py, psi, vx, vy = xT[:, 1], xT[:, 2], xT[:, 3], xT[:, 4]
        v_lat = vx * torch.sin(psi) + vy * torch.cos(psi)
        toward = torch.where(py >= 0, v_lat, -v_lat).clamp(min=0.0)
        d_wall = (self.half_w - self.wall_buffer) - (py.abs() + self.r_disk)
        ok = ok & (d_wall - toward ** 2 / (2.0 * self.wall_a_lat) >= 0.0)
        return ok, hmin

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
        if self.verify_steps is None or not self.backup:
            return self._filter_core(state, current_u, u_nom, obstacles, dt, plan_tail, samples, sample_costs)
        t0 = time.time()
        prev_saved = self._prev
        # 1. Screen: short backup + neural certificate (cheap) -> ranked candidate rows
        u_scr, info = self._filter_core(state, current_u, u_nom, obstacles, dt, plan_tail, samples, sample_costs)
        ctx = self._ctx
        order = ctx["order"][:self.verify_k]
        carried = None
        if self.carry and prev_saved is not None:   # previous commitment, shifted (always re-verified)
            pc = prev_saved[0]
            carried = np.concatenate([pc[1:], [[-self.a_brake, pc[-1, 1]]]], axis=0).astype(np.float32)
        if (self.cert is None and self.backup_steps == self.verify_steps and self.backend == "numpy"
                and (order or carried is not None)):
            # 2'. Screening already WAS the exact full-stop check (no network, same backup length): reuse its
            #     clearances instead of re-simulating. Rows: [0, n0) mode 0, [n0, 2n0) mode 1, then carried.
            clr, sf, n0 = ctx["clear"], ctx["safe"], ctx["n0"]
            for i0 in order:
                twins = [i0, i0 + n0 if i0 < n0 else i0 - n0] if i0 < 2 * n0 else [i0]
                i = next((t for t in twins if sf[t] and clr[t] >= self.verify_margin), None)
                if i is not None:
                    j = i % n0 if i < 2 * n0 else i
                    mode = 0.0 if i < n0 else (1.0 if i < 2 * n0 else float(prev_saved[1]))
                    self._prev = (ctx["cmds"][i].copy(), mode)
                    is_nom = j == 0 and plan_tail is not None
                    u_safe = ctx["nom"] if is_nom else ctx["cmds"][i, 0]
                    info.update(qp_status="nominal_verified" if is_nom else "verified_alternative", qp_active=not is_nom,
                                verify_rank=order.index(i0), dev_from_nom=np.asarray(u_safe, np.float32) - ctx["nom"],
                                latency_ms=(time.time() - t0) * 1000.0)
                    return np.asarray(u_safe, dtype=np.float32), info
            if carried is not None and len(sf) > 2 * n0 and sf[2 * n0]:     # carried plan: clear > 0 suffices
                self._prev = (ctx["cmds"][2 * n0].copy(), float(prev_saved[1]))
                u_safe = ctx["cmds"][2 * n0, 0]
                info.update(qp_status="carried_plan", qp_active=True, dev_from_nom=u_safe - ctx["nom"],
                            latency_ms=(time.time() - t0) * 1000.0)
                return np.asarray(u_safe, dtype=np.float32), info
            order, carried = [], None   # nothing passed: go straight to the exhaustive/emergency path
        if order or carried is not None:
            # 2. Verify the top-k with the exact full-length brake-to-stop backup (both backup modes)
            seqs = ctx["cmds"][order] if order else np.zeros((0, self.H, 2), dtype=np.float32)
            if carried is not None:
                seqs = np.concatenate([seqs, carried[None]], axis=0)
            k = len(seqs)
            cm = torch.as_tensor(np.concatenate([seqs, seqs], axis=0), device=self.device)
            bm = torch.cat([torch.zeros(k, device=self.device), torch.ones(k, device=self.device)])
            bs = self.backup_steps
            self.backup_steps = self.verify_steps
            traj = self._rollout(state, current_u, cm, backup_mode=bm)
            self.backup_steps = bs
            clr = self._clearance(traj, obstacles)
            need = torch.full_like(clr, float(self.verify_margin))
            if carried is not None:   # the carried plan only has to stay clear; new commitments need margin
                need[k - 1] = need[2 * k - 1] = 0.0
            ok = clr > need
            if self.max_heading is not None:
                lim = max(self.max_heading, abs(float(state[2])) + 0.05)
                ok = ok & (traj[..., 2].abs().amax(dim=1) <= lim)
            ok = ok.cpu().numpy()
            okk = ok[:k] | ok[k:]
            if okk.any():
                j = int(np.argmax(okk))
                row = order[j] if j < len(order) else -1
                self._prev = (seqs[j].copy(), 0.0 if ok[j] else 1.0)
                is_nom = row in (0, ctx["n0"]) and plan_tail is not None
                u_safe = ctx["nom"] if is_nom else seqs[j, 0]
                info["qp_status"] = "nominal_verified" if is_nom else ("carried_plan" if row == -1 else "verified_alternative")
                info["qp_active"] = not is_nom
                info["verify_rank"] = j
                info["dev_from_nom"] = np.asarray(u_safe, dtype=np.float32) - ctx["nom"]
                info["latency_ms"] = (time.time() - t0) * 1000.0
                return np.asarray(u_safe, dtype=np.float32), info
        # 3. Nothing verified: exact evaluation of all candidates (v6 behaviour)
        self._prev = prev_saved
        cert, bs = self.cert, self.backup_steps
        self.cert, self.backup_steps = None, self.verify_steps
        try:
            u_safe, info = self._filter_core(state, current_u, u_nom, obstacles, dt, plan_tail, samples, sample_costs)
        finally:
            self.cert, self.backup_steps = cert, bs
        if self.no_safe_fallback == "backup" and info["qp_status"] == "no_safe_max_clearance":
            # 4. Even the exhaustive exact check found nothing safe: execute the recovery the backups assume
            #    (yaw-damping brake) instead of the "least bad" plan, which may keep accelerating.
            p = self.p
            d_bk = float(np.clip(-(self.backup_yaw_gain or 0.0) * float(state[5]), -p.delta_max, p.delta_max))
            u_safe = np.array([-self.a_brake if float(state[3]) > 0.3 else 0.0, d_bk], dtype=np.float32)
            self._prev = None
            info["qp_status"] = "emergency_backup"
            info["dev_from_nom"] = u_safe - ctx["nom"]
        else:
            info["qp_status"] = "fallback_full:" + info["qp_status"]
        info["qp_active"] = True
        info["latency_ms"] = (time.time() - t0) * 1000.0
        return u_safe, info

    def _filter_core(self, state, current_u, u_nom, obstacles, dt, plan_tail, samples, sample_costs):
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
            if self.carry and self._prev is not None:
                # Last committed sequence shifted by one step; its final commit step starts the brake early.
                pc, pm = self._prev
                shifted = np.concatenate([pc[1:], [[-self.a_brake, pc[-1, 1]]]], axis=0).astype(np.float32)
                cmds = np.concatenate([cmds, shifted[None]], axis=0)
                cmds_t = torch.cat([cmds_t, torch.as_tensor(shifted[None], device=self.device)], dim=0)
                bmode = torch.cat([bmode, torch.full((1,), float(pm), device=self.device)])
            traj = self._rollout(state, current_u, cmds_t, backup_mode=bmode)
        else:
            traj = self._rollout(state, current_u, cmds_t)
        clear = self._clearance(traj, obstacles)
        safe = (clear > 0.0) & self._terminal_ok(traj, obstacles)
        n_cert_fail = 0
        if self.cert is not None:
            cert_ok, _ = self._certified(traj, obstacles)
            n_cert_fail = int(((clear > 0.0) & ~cert_ok).sum().item())
            safe = safe & cert_ok
        if self.max_heading is not None:
            lim = max(self.max_heading, abs(float(state[2])) + 0.05)
            heading_ok = traj[..., 2].abs().amax(dim=1) <= lim
            safe = safe & heading_ok
            clear = torch.where(heading_ok, clear, clear - 10.0)   # fallback also prefers non-spinning plans
        safe_np, clear_np = safe.cpu().numpy(), clear.cpu().numpy()

        # Full ranking of candidate rows (used by screen-then-verify): nominal first if it passed, then robust
        # candidates by progress score, then the remaining safe ones by clearance; one row per sequence.
        if self.backup:
            tr_all = traj.cpu().numpy()
            dev_all = (cmds[:, 0, 0] - nom[0]) ** 2 / 4.0 + (cmds[:, 0, 1] - nom[1]) ** 2 / 0.01
            score = -(tr_all[:, self.H - 1, 0] - float(state[0])) + 0.5 * np.abs(tr_all[:, self.H - 1, 1]) \
                + self.w_dev * dev_all
            rank = np.full(len(safe_np), np.inf)
            rob = safe_np & (clear_np >= self.pref_clear)
            rank[rob] = score[rob]
            rank[safe_np & ~rob] = 1000.0 - clear_np[safe_np & ~rob]
            if plan_tail is not None and (safe_np[0] or safe_np[n0]):
                rank[0] = rank[n0] = -1e9
            seen, order = set(), []
            for i in np.argsort(rank):
                if not np.isfinite(rank[i]):
                    break
                key = i % n0 if i < 2 * n0 else i
                if key not in seen:
                    seen.add(key)
                    order.append(int(i))
            self._ctx = {"cmds": cmds, "order": order, "n0": n0, "nom": nom, "clear": clear_np, "safe": safe_np}

        # Index 0 = nominal first action + family 0 (MPPI's own plan when provided)
        if self.backup:   # nominal is acceptable if EITHER of its backups is clear
            safe_np[0] = safe_np[0] or safe_np[n0]
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

        if self.backup:   # remember what was committed (nominal = MPPI sequence with its better backup)
            if idx == 0 and not bool(safe.cpu().numpy()[0]):
                idx = n0
            self._prev = (cmds[idx].copy(), float(bmode[idx].item()))
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
            "n_safe": int(safe_np.sum()), "n_cand": int(len(safe_np)), "n_cert_fail": n_cert_fail,
            "dev_from_nom": dev,
            "latency_ms": (time.time() - t0) * 1000.0,
        }
        return u_safe, info
