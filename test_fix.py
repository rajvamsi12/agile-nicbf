import sys, os, time
sys.path.insert(0, '/home/raj/agile_nicbf')
import numpy as np
import torch
from dynamic_bicycle_env import ClutteredRacingEnv
from mppi_planner import MPPIPlanner
from safe_controller import NeuralICBFController

class RobustSafeController(NeuralICBFController):
    def filter_control(self, state, current_u, u_nom, obstacles, dt=0.02, track_half_width=None):
        t0 = time.time()
        p = self.p
        px, py, psi, vx, vy, r = [float(s) for s in state]
        u_nom = np.asarray(u_nom, dtype=np.float64)
        nom_ax = float(np.clip(u_nom[0], -p.a_max, p.a_max))
        nom_delta = float(np.clip(u_nom[1], -p.delta_max, p.delta_max))
        u_nom_c = np.array([nom_ax, nom_delta], dtype=np.float64)

        current_delta = float(current_u[1]) if len(current_u) > 1 else nom_delta
        max_d_delta = float(p.sv_max * dt)
        delta_lo = max(-p.delta_max, current_delta - max_d_delta)
        delta_hi = min(p.delta_max, current_delta + max_d_delta)
        cand_ax = nom_ax
        cand_delta = float(np.clip(nom_delta, delta_lo, delta_hi))
        u_cand = np.array([cand_ax, cand_delta], dtype=np.float32)

        ax_lb, ax_ub = -p.a_max - cand_ax, p.a_max - cand_ax
        dl_lb, dl_ub = delta_lo - cand_delta, delta_hi - cand_delta

        def reach(nv):
            return max(nv[0] * ax_lb, nv[0] * ax_ub) + max(nv[1] * dl_lb, nv[1] * dl_ub)

        obs_rows = []
        grad_norms = []
        min_h = None
        n_no_auth = 0
        n_unreach = 0
        brake_needed = False
        has_body_disks = hasattr(p, 'lr') and hasattr(p, 'lf')
        lr = p.lr if has_body_disks else 0.0
        lf = p.lf if has_body_disks else 0.0
        hazard_horizon = min(4.5, max(1.5, 0.6 * vx)) + lf
        cos_p, sin_p = np.cos(psi), np.sin(psi)
        
        disk_offsets = (-lr, 0.0, lf) if has_body_disks else (0.0,)
        
        for obs in obstacles:
            ox, oy, r_obs = obs
            dx_body = (ox - px) * cos_p + (oy - py) * sin_p
            dy_body = -(ox - px) * sin_p + (oy - py) * cos_p
            r_eff = r_obs + self.r_margin
            if not (-r_eff - lr <= dx_body <= hazard_horizon and abs(dy_body) < (r_eff + 0.60)):
                continue
                
            for l_off in disk_offsets:
                try:
                    if self.mode == 'neural_icbf':
                        h_val, margin, p_obs = self.compute_neural_icbf_gradient(state, u_cand, obs, l_off=l_off)
                    else:
                        h_val, margin, p_obs = self.compute_obstacle_hocbf_gradient(state, u_cand, obs, l_off=l_off)
                except TypeError:
                    if self.mode == 'neural_icbf':
                        h_val, margin, p_obs = self.compute_neural_icbf_gradient(state, u_cand, obs)
                    else:
                        h_val, margin, p_obs = self.compute_obstacle_hocbf_gradient(state, u_cand, obs)
                h_val, margin = float(h_val), float(margin)
                p_obs = np.asarray(p_obs, dtype=np.float64)[:2]
                min_h = h_val if min_h is None else min(min_h, h_val)
                gn = float(np.linalg.norm(p_obs))
                grad_norms.append(gn)
                if gn < self.grad_floor:
                    if margin < 0.0:
                        n_no_auth += 1
                        brake_needed = True
                    continue
                n_vec = p_obs / gn
                rhs = margin / gn
                if rhs < 0.0 and (-rhs) > reach(n_vec) + 1e-9:
                    n_unreach += 1
                obs_rows.append((n_vec, rhs))

        # Wall rows with soft slack
        wall_rows = []
        W, aw = self.wall_limit, self.alpha_w
        wall_margins = []
        top_margins = []
        bot_margins = []
        wall_clearances = []
        
        # Enforce on CG and front disk (positive steering authority)
        wall_disk_offsets = (0.0, lf) if has_body_disks else (0.0,)
        
        for l_off in wall_disk_offsets:
            try:
                cy_dot, cy_ddot, p_wall = self.compute_wall_hocbf_gradient(state, u_cand, l_off=l_off)
            except TypeError:
                cy_dot, cy_ddot, p_wall = self.compute_wall_hocbf_gradient(state, u_cand)
                
            p_wall = np.asarray(p_wall, dtype=np.float64)[:2]
            cy = py + l_off * np.sin(psi)
            
            top_margin = -cy_ddot - 2.0 * aw * cy_dot + (aw ** 2) * (W - cy)
            bot_margin = cy_ddot + 2.0 * aw * cy_dot + (aw ** 2) * (cy + W)
            
            wall_rows.append((p_wall.copy(), float(top_margin)))
            wall_rows.append((-p_wall.copy(), float(bot_margin)))
            top_margins.append(float(top_margin))
            bot_margins.append(float(bot_margin))
            wall_margins.append(float(top_margin))
            wall_margins.append(float(bot_margin))
            wall_clearances.append(float(min(W - cy, cy + W)))
            
        wall_clearance = float(min(wall_clearances))
        min_wall_margin = float(min(wall_margins))

        any_violated = brake_needed or min_wall_margin < 0.0 or any(rhs < 0.0 for _, rhs in obs_rows)

        def _finish(u_safe, du, status, fallback, slack_vals):
            u_safe = np.array(u_safe, dtype=np.float64)
            u_safe[0] = np.clip(u_safe[0], -p.a_max, p.a_max)
            u_safe[1] = np.clip(u_safe[1], -p.delta_max, p.delta_max)
            dev_qp = u_safe - np.asarray(u_cand, dtype=np.float64)
            qp_active = bool(fallback or abs(dev_qp[0]) > self.tol_ax or abs(dev_qp[1]) > self.tol_delta)
            info = {
                'qp_active': qp_active,
                'dev_from_nom': u_safe - u_nom_c,
                'v_star': np.asarray(du, dtype=np.float64),
                'min_h_val': float(min_h) if min_h is not None else 1.0,
                'wall_clearance_m': wall_clearance,
                'wall_margin_min': float(min_wall_margin),
                'grad_norm_min': float(min(grad_norms)) if grad_norms else None,
                'n_obstacle_rows': len(obs_rows),
                'n_no_authority': n_no_auth,
                'n_unreachable': n_unreach,
                'brake_override': bool(brake_needed),
                'slack_vals': [float(s) for s in slack_vals],
                'slack_max': float(max(slack_vals)) if len(slack_vals) else 0.0,
                'qp_status': status,
                'wall_fallback': bool(fallback),
                'latency_ms': (time.time() - t0) * 1000.0,
            }
            return u_safe.astype(np.float32), info

        if not any_violated:
            return _finish(u_cand, np.zeros(2), 'skipped_all_satisfied', False, [])

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
            row[2 + m_obs + k] = -1.0
            G.append(row)
            h.append(mg / s)
            
        P_diag = np.array([2.0, 0.1] + [self.w_slack] * m_obs + [self.w_slack * 5.0] * m_wall, dtype=np.float64)
        q_vec = np.array([0.0, 0.0] + [self.w_slack_lin] * m_obs + [self.w_slack_lin * 5.0] * m_wall, dtype=np.float64)
        lb = np.array([ax_lb, dl_lb] + [0.0] * m_total, dtype=np.float64)
        ub = np.array([ax_ub, dl_ub] + [1e6] * m_total, dtype=np.float64)

        if brake_needed and getattr(self, 'brake_clamp', True):
            ub[0] = min(ub[0], min(-self.brake_decel - cand_ax, 0.0))

        z, status = self._solve_qp(P_diag, q_vec, np.array(G), np.array(h), lb, ub)

        if z is not None:
            du = z[:2]
            return _finish(np.asarray(u_cand, dtype=np.float64) + du, du, status, False, z[2:])

        if min(top_margins) < min(bot_margins) or (min(top_margins) == min(bot_margins) and py > 0):
            direction = -np.sign(p_wall[1]) if abs(p_wall[1]) > 1e-9 else -1.0
        else:
            direction = np.sign(p_wall[1]) if abs(p_wall[1]) > 1e-9 else 1.0
        delta_fb = delta_hi if direction > 0 else delta_lo
        ax_fb = min(cand_ax, -min(p.a_max, self.brake_decel))
        u_fb = np.array([ax_fb, delta_fb], dtype=np.float64)
        return _finish(u_fb, u_fb - np.asarray(u_cand, dtype=np.float64), status, True, [])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seeds = [101, 138, 175, 212, 249]

for mu in [1.0, 0.7]:
    print(f'=== TESTING MU = {mu} ===')
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = ClutteredRacingEnv(num_obstacles=10, track_length=50.0, track_width=4.0, seed=seed)
        env.model.p.mu = float(mu)
        state = env.reset(start_vx=3.0)
        
        planner = MPPIPlanner(num_samples=50, horizon=30, v_target=6.0, r_margin=0.26, device=device)
        shield = RobustSafeController(model_path='/home/raj/agile_nicbf/models/neural_icbf_6d.pth', mode='neural_icbf', r_margin=0.26, alpha1=2.5, alpha2=3.5, alpha_w=2.5, device=device)
        
        current_u = np.array([env.actual_ax, env.actual_delta], dtype=np.float32)
        speeds, stall_run = [], 0
        outcome, step = 'timeout', 0
        
        for step in range(1, 2001):
            perceived = env.get_perceived_obstacles()
            obs_state = env.get_noisy_state()
            u_nom = planner.plan(obs_state, perceived, barrier_net=shield.barrier_net)
            u_exec, qi = shield.filter_control(obs_state, current_u, u_nom, perceived, dt=0.02)
            state, info = env.step(u_exec, dt=0.02)
            current_u = info['actual_u']
            speeds.append(float(state[3]))
            stall_run = stall_run + 1 if state[3] < 0.5 else 0
            
            if info['collision']: outcome = 'collision'; break
            if info['out_of_bounds']: outcome = 'out_of_bounds'; break
            if info['success']: outcome = 'success'; break
            if stall_run > 150: outcome = 'stalled'; break
            
        print(f'Seed {seed}: outcome={outcome}, steps={step}, time={step*0.02:.2f}s, mean_speed={np.mean(speeds):.2f} m/s, final_x={state[0]:.2f}')
