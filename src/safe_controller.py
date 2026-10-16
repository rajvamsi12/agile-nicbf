import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from qpsolvers import solve_qp

# Universal import handling
try:
    from .dynamic_bicycle_env import DynamicBicycleModel, DynamicBicycleParams, ClutteredRacingEnv
    from .train_neural_icbf import NeuralBarrierNet
    from .mppi_planner import MPPIPlanner
except ImportError:
    from dynamic_bicycle_env import DynamicBicycleModel, DynamicBicycleParams, ClutteredRacingEnv
    from train_neural_icbf import NeuralBarrierNet
    from mppi_planner import MPPIPlanner


class NeuralICBFController:
    """
    Millisecond-scale Neural Integral Control Barrier Function (NICBF) Safety Shield.
    Enforces learned/geometric safety constraints, actuator bounds, and relative-degree-2 HOCBF track boundaries and obstacles.
    References: Ames et al. (ICBF, 2020), Xiao & Belta (HOCBF, IEEE TAC 2022).
    """
    def __init__(self,
                 model_path=None,
                 mode='neural_icbf',  # 'neural_icbf' (learned barrier net) or 'analytical_hocbf' (baseline)
                 alpha_nicbf=3.0,     # Class-K gain for Neural ICBF (h_dot + alpha * h >= 0)
                 alpha1=2.5,          # Relative-degree-2 barrier gain 1 (analytical HOCBF)
                 alpha2=3.5,          # Relative-degree-2 barrier gain 2 (analytical HOCBF)
                 alpha_w=3.5,         # Track wall HOCBF gain
                 gamma_u=None,        # DEPRECATED/unused: input-ellipse row was removed (box bounds enforce limits)
                 r_margin=0.38,       # Safety boundary margin (0.21m disk + 0.05m buffer)
                 w_slack=1000.0,      # Quadratic per-row obstacle slack weight (normalized control units)
                 w_slack_lin=100.0,   # LINEAR slack penalty -> exact penalty: slack is 0 whenever the row is attainable
                 solver='quadprog',   # Exact active-set solver
                 wall_limit=1.75,     # |py| limit for the CG: 2.0 wall - 0.155 half-width - 0.095 buffer
                 grad_floor=1e-3,     # ||p|| below this => constraint has no steering/throttle authority
                 brake_decel=4.0,     # m/s^2 used by brake override and wall fallback
                 tol_ax=0.05,         # m/s^2   intervention threshold (QP vs slew-projected candidate)
                 tol_delta=0.005,     # rad     intervention threshold
                 brake_clamp=True,    # If False, disables hard clamp on ub[0] when brake_needed is triggered
                 disable_wall_cbf=False, # If True, disables track wall safety barrier constraints
                 wall_steer_guard='off', # 'off' | 'zero_steer' | 'drop_row': handling of a wall row whose steering
                                         # authority d(cy_ddot)/d(delta) is inverted (tire saturation, e.g. a spin)
                 obs_steer_guard=False,  # If True, zero the steering component of obstacle rows whose steering
                                         # authority disagrees in sign with the kinematic (linear-tire) expectation
                 slip_brake_limit=None,  # m/s^2 (e.g. 1.0): when rear slip or yaw rate is high, the QP may not brake
                                         # harder than this beyond the planner's own command (brake-induced oversteer)
                 device='cpu'):
        
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = DynamicBicycleModel(device=self.device)
        self.p = self.model.p
        
        self.mode = mode
        self.alpha_nicbf = alpha_nicbf
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha_w = alpha_w
        self.r_margin = r_margin
        self.w_slack = w_slack
        self.w_slack_lin = w_slack_lin
        self.solver = solver
        self.wall_limit = wall_limit
        self.grad_floor = grad_floor
        self.brake_decel = brake_decel
        self.tol_ax = tol_ax
        self.tol_delta = tol_delta
        self.brake_clamp = brake_clamp
        self.disable_wall_cbf = disable_wall_cbf
        assert wall_steer_guard in ("off", "zero_steer", "drop_row"), wall_steer_guard
        self.wall_steer_guard = wall_steer_guard
        self.obs_steer_guard = obs_steer_guard
        self.slip_brake_limit = slip_brake_limit
        
        # Load trained Neural Barrier Network
        self.barrier_net = NeuralBarrierNet(input_dim=6, hidden_dim=256).to(self.device)
        
        # Robust model path resolution
        resolved_path = model_path
        if resolved_path is None or not os.path.exists(resolved_path):
            candidates = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "neural_icbf_6d.pth"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "neural_icbf_6d.pth"),
                os.environ.get("NICBF_MODEL_PATH", ""),
                os.path.abspath("models/neural_icbf_6d.pth"),
            ]
            for cand in candidates:
                if cand and os.path.exists(cand):
                    resolved_path = cand
                    break
                    
        if resolved_path is not None and os.path.exists(resolved_path):
            self.barrier_net.load_state_dict(torch.load(resolved_path, map_location=self.device, weights_only=False))
            self.barrier_net.eval()
            print(f"[Stage 4] Loaded Neural Barrier Network from {resolved_path} (mode={self.mode})")
        else:
            print(f"[Stage 4] Warning: Model checkpoint not found. Running with uninitialized weights.")

    def compute_neural_icbf_gradient(self, state, u_cand, obs, l_off=0.0):
        """
        Computes 2nd-order Lie derivatives and control Jacobian using the trained 6D Neural Barrier Network
        evaluated at body-center offset l_off.
        """
        px, py, psi, vx, vy, r = state
        ox, oy, ro = obs
        c, s = np.cos(psi), np.sin(psi)
        cx, cy = px + l_off * c, py + l_off * s
        dx = cx - ox
        dy = cy - oy
        r_eff = ro + self.r_margin
        scale = 0.60 / r_eff
        
        x_rel = torch.tensor([[dx * scale, dy * scale, psi, vx, vy, r]], dtype=torch.float32, requires_grad=True, device=self.device)
        h_val = self.barrier_net(x_rel).squeeze()
        grad_x = torch.autograd.grad(h_val, x_rel, create_graph=True)[0].squeeze()
        
        u_t = torch.as_tensor(u_cand, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_()
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        f = self.model.continuous_dynamics_torch(s_t, u_t).squeeze()
        
        px_dot = vx * c - vy * s
        py_dot = vx * s + vy * c
        cx_dot = px_dot - l_off * r * s
        cy_dot = py_dot + l_off * r * c
        
        h_dot = grad_x[0] * cx_dot * scale + grad_x[1] * cy_dot * scale + grad_x[2] * r + grad_x[3] * f[3] + grad_x[4] * f[4] + grad_x[5] * f[5]
        
        px_ddot = (f[3] - vy * r) * torch.cos(s_t[0, 2]) - (f[4] + vx * r) * torch.sin(s_t[0, 2])
        py_ddot = (f[3] - vy * r) * torch.sin(s_t[0, 2]) + (f[4] + vx * r) * torch.cos(s_t[0, 2])
        cx_ddot = px_ddot - l_off * (f[5] * s + r**2 * c)
        cy_ddot = py_ddot + l_off * (f[5] * c - r**2 * s)
        
        h_ddot_spatial = grad_x[0] * cx_ddot * scale + grad_x[1] * cy_ddot * scale
        
        hocbf_margin = h_ddot_spatial + (self.alpha1 + self.alpha2) * h_dot + (self.alpha1 * self.alpha2) * h_val
        p_obs = torch.autograd.grad(hocbf_margin, u_t, create_graph=False)[0].squeeze().detach().cpu().numpy().astype(np.float64)
        return h_val.item(), hocbf_margin.item(), p_obs

    def compute_obstacle_hocbf_gradient(self, state, u_cand, obs, l_off=0.0):
        """
        Computes exact 2nd-order Lie derivatives and control Jacobian for analytical geometric obstacle avoidance
        evaluated at body-center offset l_off.
        Relative degree = 2: position -> velocity -> acceleration -> control u.
        Condition: h_ddot + (alpha1 + alpha2)*h_dot + alpha1*alpha2*h >= 0
        """
        px, py, psi, vx, vy, r = state
        ox, oy, ro = obs
        c, s = np.cos(psi), np.sin(psi)
        cx, cy = px + l_off * c, py + l_off * s
        dx, dy = cx - ox, cy - oy
        R = ro + self.r_margin
        h = dx**2 + dy**2 - R**2
        px_dot = vx * c - vy * s
        py_dot = vx * s + vy * c
        cx_dot = px_dot - l_off * r * s
        cy_dot = py_dot + l_off * r * c
        h_dot = 2 * dx * cx_dot + 2 * dy * cy_dot

        u_t = torch.as_tensor(u_cand, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_()
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        f = self.model.continuous_dynamics_torch(s_t, u_t).squeeze()
        px_ddot = (f[3] - vy * r) * torch.cos(s_t[0, 2]) - (f[4] + vx * r) * torch.sin(s_t[0, 2])
        py_ddot = (f[3] - vy * r) * torch.sin(s_t[0, 2]) + (f[4] + vx * r) * torch.cos(s_t[0, 2])
        cx_ddot = px_ddot - l_off * (f[5] * s + r**2 * c)
        cy_ddot = py_ddot + l_off * (f[5] * c - r**2 * s)

        h_ddot = 2 * (cx_dot**2 + cy_dot**2) + 2 * dx * cx_ddot + 2 * dy * cy_ddot
        margin = h_ddot + (self.alpha1 + self.alpha2) * h_dot + self.alpha1 * self.alpha2 * h
        p_obs = torch.autograd.grad(h_ddot, u_t)[0].squeeze().detach().cpu().numpy().astype(np.float64)
        return h, margin.item(), p_obs

    def compute_wall_hocbf_gradient(self, state, current_u, l_off=0.0):
        """
        Computes exact 2nd-order Lie derivatives and control Jacobian for track containment
        evaluated on a disk center at longitudinal offset l_off.
        Relative degree = 2: cy -> cy_dot -> cy_ddot(u).
        """
        px, py, psi, vx, vy, r = state
        c, s = np.cos(psi), np.sin(psi)
        
        u_tensor = torch.as_tensor(current_u, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_()
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        f = self.model.continuous_dynamics_torch(state_tensor, u_tensor).squeeze()
        py_dot = vx * s + vy * c
        cy_dot = py_dot + l_off * r * c
        
        py_ddot = (f[3] - vy * r) * torch.sin(state_tensor[0, 2]) + (f[4] + vx * r) * torch.cos(state_tensor[0, 2])
        cy_ddot = py_ddot + l_off * (f[5] * c - r**2 * s)
        
        p_wall = torch.autograd.grad(cy_ddot, u_tensor, create_graph=False)[0].squeeze().detach().cpu().numpy().astype(np.float64)
        return cy_dot, cy_ddot.item(), p_wall

    # ------------------------------------------------------------------
    # QP back-end (isolated so it can be unit-tested / swapped)
    # ------------------------------------------------------------------
    def _solve_qp(self, P_diag, q, G, h, lb, ub):
        """Returns (z, status). status in {'ok', 'infeasible', 'exception:<Type>'}."""
        try:
            z = solve_qp(np.diag(P_diag), np.asarray(q, dtype=np.float64), G, h, lb=lb, ub=ub, solver=self.solver)
        except Exception as e:  # solver raised (e.g. quadprog inconsistent constraints)
            return None, f"exception:{type(e).__name__}"
        if z is None:
            return None, "infeasible"
        return np.asarray(z, dtype=np.float64), "ok"

    def filter_control(self, state, current_u, u_nom, obstacles, dt=0.02, track_half_width=None):
        """
        Minimal-intervention CBF-QP safety filter over 3-disk footprint (rear, CG, front).
        """
        t0 = time.time()
        p = self.p
        px, py, psi, vx, vy, r = [float(s) for s in state]
        cos_p, sin_p = np.cos(psi), np.sin(psi)
        
        # Actuator Delay Compensation (Drivetrain tau_motor = 0.05s)
        # Propagate state forward by tau_delay using RK4 integration with current_u to eliminate phase lag
        tau_delay = getattr(self, "tau_delay", 0.05)
        if tau_delay > 0:
            pred_state = self.model.step_rk4_numpy(state, current_u, dt=tau_delay)
            px_p, py_p, psi_p, vx_p, vy_p, r_p = [float(s) for s in pred_state]
            cos_pp, sin_pp = np.cos(psi_p), np.sin(psi_p)
            eval_state = pred_state
        else:
            eval_state = state
            px_p, py_p, psi_p, vx_p, vy_p, r_p = px, py, psi, vx, vy, r
            cos_pp, sin_pp = cos_p, sin_p
        u_nom = np.asarray(u_nom, dtype=np.float64)
        nom_ax = float(np.clip(u_nom[0], -p.a_max, p.a_max))
        nom_delta = float(np.clip(u_nom[1], -p.delta_max, p.delta_max))
        u_nom_c = np.array([nom_ax, nom_delta], dtype=np.float64)

        # 1. Slew-consistent candidate (steering projected into the 1-step reachable window)
        current_delta = float(current_u[1]) if len(current_u) > 1 else nom_delta
        max_d_delta = float(p.sv_max * dt)
        delta_lo = max(-p.delta_max, current_delta - max_d_delta)
        delta_hi = min(p.delta_max, current_delta + max_d_delta)
        cand_ax = nom_ax
        cand_delta = float(np.clip(nom_delta, delta_lo, delta_hi))

        # Active Escape Steering Bias: only if slow, pointed straight, and facing an obstacle
        if vx < 1.5 and abs(psi) < 0.35:
            # check if nearest obstacle is within 2.5m directly ahead
            min_dist_head = 999.0
            chosen_dir = 0.0
            for obs in obstacles:
                ox, oy, ro = obs
                dx_b = (ox - px) * cos_p + (oy - py) * sin_p
                dy_b = -(ox - px) * sin_p + (oy - py) * cos_p
                if 0.0 < dx_b < 2.5 and abs(dy_b) < (ro + self.r_margin + 0.15):
                    if dx_b < min_dist_head:
                        min_dist_head = dx_b
                        # Steer toward the center of the track (away from wall and obstacle)
                        chosen_dir = -1.0 if py > 0 else 1.0
            if min_dist_head < 999.0:
                escape_delta = float(chosen_dir * 0.25)
                cand_delta = float(np.clip(escape_delta, delta_lo, delta_hi))

        u_cand = np.array([cand_ax, cand_delta], dtype=np.float32)

        # Box on du
        ax_lb, ax_ub = max(-p.a_max, -2.5) - cand_ax, p.a_max - cand_ax
        dl_lb, dl_ub = delta_lo - cand_delta, delta_hi - cand_delta

        def reach(nv):  # max of n^T du over the box
            return max(nv[0] * ax_lb, nv[0] * ax_ub) + max(nv[1] * dl_lb, nv[1] * dl_ub)

        # 2. Obstacle rows over 3 disks (rear axle -lr, CG 0, front axle +lf)
        obs_rows = []              # (n_vec, rhs)  with rhs = margin/||p||
        grad_norms = []
        min_h = None
        n_no_auth = 0
        n_unreach = 0
        brake_needed = False
        has_body_disks = hasattr(p, 'lr') and hasattr(p, 'lf')
        lr = p.lr if has_body_disks else 0.0
        lf = p.lf if has_body_disks else 0.0
        hazard_horizon = min(4.5, max(2.0, 0.65 * vx_p)) + lf
        cos_p, sin_p = np.cos(psi), np.sin(psi)
        
        disk_offsets = (0.0,)
        
        for obs in obstacles:
            ox, oy, r_obs = obs
            dx_body = (ox - px_p) * cos_pp + (oy - py_p) * sin_pp
            dy_body = -(ox - px_p) * sin_pp + (oy - py_p) * cos_pp
            r_eff = r_obs + self.r_margin
            # Track-corridor aware hazard filter:
            # 1. Body-frame forward lookahead
            if not (-r_eff <= dx_body <= hazard_horizon and abs(dy_body) < (r_eff + 0.35)):
                continue
            # 2. Heading-divergence protection:
            # If the vehicle is already angled away from the obstacle by > 12 degrees
            # and moving laterally away, do not force further steering that would cause track spinout
            dy_track = oy - py
            if (dy_body < 0 and psi > 0.20) or (dy_body > 0 and psi < -0.20):
                continue
            if abs(dy_track) > (r_eff + 0.20) and ((dy_track < 0 and nom_delta > 0) or (dy_track > 0 and nom_delta < 0)):
                continue
                continue
                
            for l_off in disk_offsets:
                try:
                    if self.mode == 'neural_icbf':
                        h_val, margin, p_obs = self.compute_neural_icbf_gradient(eval_state, u_cand, obs, l_off=l_off)
                    else:
                        h_val, margin, p_obs = self.compute_obstacle_hocbf_gradient(eval_state, u_cand, obs, l_off=l_off)
                except TypeError:
                    if self.mode == 'neural_icbf':
                        h_val, margin, p_obs = self.compute_neural_icbf_gradient(eval_state, u_cand, obs)
                    else:
                        h_val, margin, p_obs = self.compute_obstacle_hocbf_gradient(eval_state, u_cand, obs)
                h_val, margin = float(h_val), float(margin)
                p_obs = np.asarray(p_obs, dtype=np.float64)[:2]
                if self.obs_steer_guard:
                    # Linear-tire expectation: +delta accelerates the body to its left, so d(h_ddot)/d(delta) has the
                    # sign of the obstacle-to-car vector projected on the body-left axis. Disagreement = saturated tire.
                    left_proj = -(px_p - ox) * sin_pp + (py_p - oy) * cos_pp
                    if p_obs[1] * left_proj < 0.0:
                        p_obs[1] = 0.0
                min_h = h_val if min_h is None else min(min_h, h_val)
                gn = float(np.linalg.norm(p_obs))
                grad_norms.append(gn)
                if gn < self.grad_floor:
                    if margin < 0.0:       # violated but cannot be fixed through u -> brake
                        n_no_auth += 1
                        brake_needed = True
                    continue
                n_vec = p_obs / gn
                rhs = margin / gn
                if rhs < 0.0 and (-rhs) > reach(n_vec) + 1e-9:
                    n_unreach += 1         # 1-step slew cannot fully close gap; slack absorbs difference
                obs_rows.append((n_vec, rhs))

        # 3. Dynamic Envelope Wall Containment at CG (eliminates opposing rear-bumper torque conflict)
        # Safety envelope dynamically shrinks allowed CG corridor based on vehicle yaw:
        # body_extent = half_width * |cos(psi)| + max(lf, lr) * |sin(psi)| + buffer
        wall_rows = []
        track_W = 2.0 if track_half_width is None else float(track_half_width)
        body_extent = 0.21 + max(lf, lr) * abs(sin_pp) + 0.10
        W_eff = max(1.40, track_W - body_extent)
        aw = self.alpha_w
        wall_steer_inverted = False
        
        if getattr(self, "disable_wall_cbf", False):
            wall_rows = []
            top_margin = 1.0
            bot_margin = 1.0
            wall_margins = [1.0, 1.0]
            wall_clearance = 1.0
            min_wall_margin = 1.0
        else:
            # Enforced at vehicle center of mass (l_off = 0.0) where steering authority is strictly positive
            cy_dot, cy_ddot, p_wall = self.compute_wall_hocbf_gradient(eval_state, u_cand, l_off=0.0)
            p_wall = np.asarray(p_wall, dtype=np.float64)[:2]
            
            top_margin = -cy_ddot - 2.0 * aw * cy_dot + (aw ** 2) * (W_eff - py_p)
            bot_margin = cy_ddot + 2.0 * aw * cy_dot + (aw ** 2) * (py_p + W_eff)
            
            # Steering-authority sign guard: in the linear-tire regime, steering left (+delta) raises the
            # lateral acceleration toward +y, so d(cy_ddot)/d(delta) > 0. Past front-tire saturation (a spin),
            # only the Fyf*sin/cos(delta) projection terms remain and the sign inverts; the linearized row then
            # demands steering INTO the wall it protects and overrides MPPI's counter-steer recovery.
            wall_steer_inverted = bool(p_wall[1] <= 0.0)
            if wall_steer_inverted and self.wall_steer_guard == "zero_steer":
                p_wall[1] = 0.0                     # keep braking authority, remove the inverted steering one
            if not (wall_steer_inverted and self.wall_steer_guard == "drop_row"):
                wall_rows.append((p_wall.copy(), float(top_margin)))
                wall_rows.append((-p_wall.copy(), float(bot_margin)))
            else:                                   # linearization invalid: defer to MPPI's nonlinear rollouts
                top_margin, bot_margin = max(top_margin, 0.0), max(bot_margin, 0.0)
            top_margins = [float(top_margin)]
            bot_margins = [float(bot_margin)]
            wall_margins = [float(top_margin), float(bot_margin)]
            wall_clearance = float(min(W_eff - py_p, py_p + W_eff))
            min_wall_margin = float(min(wall_margins))

        wall_top_viol = bool(top_margin < 0.0)
        wall_bot_viol = bool(bot_margin < 0.0)
        obstacle_viol = bool(any(rhs < 0.0 for _, rhs in obs_rows))
        any_violated = brake_needed or min_wall_margin < 0.0 or obstacle_viol

        def _finish(u_safe, du, status, fallback, slack_vals):
            u_safe = np.array(u_safe, dtype=np.float64)
            u_safe[0] = np.clip(u_safe[0], -p.a_max, p.a_max)
            u_safe[1] = np.clip(u_safe[1], -p.delta_max, p.delta_max)
            dev_qp = u_safe - np.asarray(u_cand, dtype=np.float64)
            qp_active = bool(fallback or abs(dev_qp[0]) > self.tol_ax or abs(dev_qp[1]) > self.tol_delta)
            info = {
                "qp_active": qp_active,                       # shield moved u away from slew-projected candidate
                "dev_from_nom": u_safe - u_nom_c,             # total deviation from planner output (incl. slew clip)
                "v_star": np.asarray(du, dtype=np.float64),
                "min_h_val": float(min_h) if min_h is not None else 1.0,
                "wall_clearance_m": wall_clearance,           # vs dynamic envelope limit W_eff
                "wall_margin_min": float(min_wall_margin),
                "wall_top_viol": wall_top_viol,
                "wall_bot_viol": wall_bot_viol,
                "wall_steer_inverted": wall_steer_inverted,
                "obstacle_viol": obstacle_viol,
                "brake_needed": bool(brake_needed),
                "grad_norm_min": float(min(grad_norms)) if grad_norms else None,
                "n_obstacle_rows": len(obs_rows),
                "n_no_authority": n_no_auth,
                "n_unreachable": n_unreach,
                "brake_override": bool(brake_needed),
                "slack_vals": [float(s) for s in slack_vals],
                "slack_max": float(max(slack_vals)) if len(slack_vals) else 0.0,
                "qp_status": status,
                "wall_fallback": bool(fallback),
                "latency_ms": (time.time() - t0) * 1000.0,
            }
            return u_safe.astype(np.float32), info

        # 4. Nothing violated -> du = 0 is the exact QP optimum
        if not any_violated:
            return _finish(u_cand, np.zeros(2), "skipped_all_satisfied", False, [])

        # 5. Assemble QP with soft slack on BOTH obstacles and walls
        m_obs = len(obs_rows)
        m_wall = len(wall_rows)
        m_total = m_obs + m_wall
        n = 2 + m_total
        G, h = [], []
        for k, (n_vec, rhs) in enumerate(obs_rows):
            row = np.zeros(n)
            row[0:2] = -n_vec
            row[2 + k] = -1.0
            G.append(row)
            h.append(rhs)
        for k, (pw, mg) in enumerate(wall_rows):
            s = max(float(np.linalg.norm(pw)), self.grad_floor)
            row = np.zeros(n)
            row[0:2] = pw / s
            row[2 + m_obs + k] = -1.0    # soft slack for wall boundary!
            G.append(row)
            h.append(mg / s)
            
        P_diag = np.array([2.0, 0.1] + [self.w_slack] * m_obs + [self.w_slack * 5.0] * m_wall, dtype=np.float64)
        q_vec = np.array([0.0, 0.0] + [self.w_slack_lin] * m_obs + [self.w_slack_lin * 5.0] * m_wall, dtype=np.float64)
        # Anti-stall forward crawl floor:
        # Never allow the vehicle to brake to a dead stop when speed drops below 1.2 m/s.
        if vx < 1.2:
            crawl_min = 0.6
            ax_lb = max(ax_lb, crawl_min - cand_ax)
            if ax_ub < ax_lb:
                ax_ub = ax_lb + 0.2
            brake_clamp_active = False
        else:
            brake_clamp_active = getattr(self, 'brake_clamp', True)

        if self.slip_brake_limit is not None:
            alpha_r_p = -np.arctan2(vy_p - lr * r_p, max(vx_p, p.v_min))
            if abs(alpha_r_p) > 0.10 or abs(r_p) > 2.5:   # rear tire near saturation / large yaw transient
                ax_lb = max(ax_lb, min(cand_ax, -self.slip_brake_limit) - cand_ax)
                ax_ub = max(ax_ub, ax_lb)

        lb = np.array([ax_lb, dl_lb] + [0.0] * m_total, dtype=np.float64)
        ub = np.array([ax_ub, dl_ub] + [1e6] * m_total, dtype=np.float64)
        
        # Apply brake clamp only if allowed and not crawling
        if brake_needed and brake_clamp_active:
            ub[0] = min(ub[0], min(-self.brake_decel - cand_ax, 0.0))
            if ub[0] < lb[0]:
                ub[0] = lb[0]

        z, status = self._solve_qp(P_diag, q_vec, np.array(G), np.array(h), lb, ub)

        if z is not None:
            du = z[:2]
            return _finish(np.asarray(u_cand, dtype=np.float64) + du, du, status, False, z[2:])

        # 6. Fallback (rare numerical breakdown): smoothly steer away from nearest wall without locking brakes
        direction = -np.sign(py) if abs(py) > 0.01 else -1.0
        delta_fb = delta_hi if direction > 0 else delta_lo
        if vx < 1.2:
            ax_fb = max(0.5, min(cand_ax, 1.5))
        else:
            ax_fb = max(-3.0, min(cand_ax, 0.0))
        u_fb = np.array([ax_fb, delta_fb], dtype=np.float64)
        return _finish(u_fb, u_fb - np.asarray(u_cand, dtype=np.float64), status, True, [])



# ---------------------------------------------------------------------------
# Closed-Loop Integration Test: MPPI + Neural ICBF-QP Shield
# ---------------------------------------------------------------------------
def test_agile_nicbf_closed_loop():
    print(f"\n[Stage 4] Testing Closed-Loop Agile-NICBF (MPPI + Neural ICBF Safety Shield)...")
    env = ClutteredRacingEnv(num_obstacles=20, track_length=50.0, seed=101)
    state = env.reset(start_vx=3.0)
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "neural_icbf_6d.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "neural_icbf_6d.pth")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    planner = MPPIPlanner(num_samples=100, horizon=30, v_target=6.0, r_margin=0.38, device=device)
    shield = NeuralICBFController(model_path=model_path, mode='neural_icbf', r_margin=0.38, alpha1=2.5, alpha2=3.5, alpha_w=2.5, device=device)
    
    current_u = np.array([env.actual_ax, env.actual_delta])
    total_steps = 0
    start_time = time.time()
    done = False
    
    qp_interventions = 0
    total_latency = []
    
    while not done and total_steps < 600:
        px, py = state[0], state[1]
        perceived_obs = env.get_perceived_obstacles()
        obs_state = env.get_noisy_state()
        
        t0 = time.time()
        # 1. MPPI Nominal Plan
        u_nom = planner.plan(obs_state, perceived_obs)
        # 2. Neural ICBF Multi-Obstacle QP Shield
        u_safe, qp_info = shield.filter_control(obs_state, current_u, u_nom, perceived_obs, dt=0.02)
        step_latency = (time.time() - t0) * 1000.0
        total_latency.append(step_latency)
        
        if qp_info["qp_active"]:
            qp_interventions += 1
            
        state, info = env.step(u_safe, dt=0.02)
        current_u = info["actual_u"]   # actual actuator state (servo slew / motor lag)
        total_steps += 1
        
        if info["collision"]:
            print(f"Step {total_steps:03d} | Collided at x={state[0]:.2f}m, y={state[1]:.2f}m!")
            done = True
        elif info["out_of_bounds"]:
            print(f"Step {total_steps:03d} | Out of bounds at y={state[1]:.2f}m!")
            done = True
        elif info["success"]:
            print(f"Step {total_steps:03d} | SUCCESS: Crossed finish line safely at x={state[0]:.2f}m in {total_steps*0.02:.2f}s!")
            done = True
            
    elapsed = time.time() - start_time
    avg_lat = np.mean(total_latency) if len(total_latency) > 0 else 0.0
    print(f"[Stage 4] Closed-Loop Completed: {total_steps} steps in {elapsed:.2f}s | Avg Step Latency: {avg_lat:.2f} ms | QP Interventions: {qp_interventions}/{total_steps} ({(qp_interventions/max(1, total_steps))*100:.1f}%)")

if __name__ == "__main__":
    test_agile_nicbf_closed_loop()
