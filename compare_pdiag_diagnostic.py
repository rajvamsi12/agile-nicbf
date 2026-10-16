import os
import sys
import time
import json
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dynamic_bicycle_env import ClutteredRacingEnv
from mppi_planner import MPPIPlanner
from safe_controller import NeuralICBFController

seeds = [101, 138, 175, 212, 249]
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.path.join(current_dir, "models", "neural_icbf_6d.pth")

configs = [
    {"name": "Remote_Pdiag_1_20_BrakeClamp", "p_diag": [1.0, 20.0], "brake_clamp": True},
    {"name": "Local_Pdiag_2_0.1_BrakeClamp",  "p_diag": [2.0, 0.1],  "brake_clamp": True},
    {"name": "Local_Pdiag_2_0.1_NoBrakeClamp", "p_diag": [2.0, 0.1], "brake_clamp": False},
]

methods = ["analytical_hocbf", "neural_icbf", "agile_nicbf_coupled"]

print("=" * 80)
print("  Diagnostic Evaluation: P_diag & Brake-Override Ablation on GPU")
print(f"  Device: {device} | Seeds: {seeds}")
print("=" * 80)

summary_results = {}

for cfg in configs:
    cfg_name = cfg["name"]
    p_diag_weights = cfg["p_diag"]
    enable_brake_clamp = cfg["brake_clamp"]
    
    print(f"\n>>> Running Config: {cfg_name} (P_diag={p_diag_weights}, brake_clamp={enable_brake_clamp})")
    summary_results[cfg_name] = {}
    
    for method in methods:
        shield_mode = 'neural_icbf' if method in ['neural_icbf', 'agile_nicbf_coupled'] else 'analytical_hocbf'
        
        # Instantiate shield and modify P_diag and ub[0] clamp behavior directly in a clean subclass
        class DiagnosedShield(NeuralICBFController):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.target_p_diag = p_diag_weights
                self.allow_brake_clamp = enable_brake_clamp

            def filter_control(self, state, current_u, u_nom, obstacles, dt=0.02, track_half_width=None):
                # Run standard filter control
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
                hazard_horizon = min(4.5, max(1.5, 0.6 * vx))
                cos_p, sin_p = np.cos(psi), np.sin(psi)
                for obs in obstacles:
                    ox, oy, r_obs = obs
                    dx_body = (ox - px) * cos_p + (oy - py) * sin_p
                    dy_body = -(ox - px) * sin_p + (oy - py) * cos_p
                    r_eff = r_obs + self.r_margin
                    if not (-r_eff <= dx_body <= hazard_horizon and abs(dy_body) < (r_eff + 0.60)):
                        continue
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

                py_dot, py_ddot, p_wall = self.compute_wall_hocbf_gradient(state, u_cand)
                p_wall = np.asarray(p_wall, dtype=np.float64)[:2]
                W, aw = self.wall_limit, self.alpha_w
                top_margin = -py_ddot - 2.0 * aw * py_dot + (aw ** 2) * (W - py)
                bot_margin = py_ddot + 2.0 * aw * py_dot + (aw ** 2) * (py + W)
                wall_rows = [(p_wall.copy(), float(top_margin)), (-p_wall, float(bot_margin))]
                wall_clearance = float(min(W - py, py + W))

                any_violated = brake_needed or top_margin < 0.0 or bot_margin < 0.0 or any(rhs < 0.0 for _, rhs in obs_rows)

                def _finish(u_safe, du, status, fallback, slack_vals):
                    u_safe = np.array(u_safe, dtype=np.float64)
                    u_safe[0] = np.clip(u_safe[0], -p.a_max, p.a_max)
                    u_safe[1] = np.clip(u_safe[1], -p.delta_max, p.delta_max)
                    dev_qp = u_safe - np.asarray(u_cand, dtype=np.float64)
                    qp_active = bool(fallback or abs(dev_qp[0]) > self.tol_ax or abs(dev_qp[1]) > self.tol_delta)
                    info = {
                        "qp_active": qp_active,
                        "dev_from_nom": u_safe - u_nom_c,
                        "v_star": np.asarray(du, dtype=np.float64),
                        "min_h_val": float(min_h) if min_h is not None else 1.0,
                        "wall_clearance_m": wall_clearance,
                        "wall_margin_min": float(min(top_margin, bot_margin)),
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

                if not any_violated:
                    return _finish(u_cand, np.zeros(2), "skipped_all_satisfied", False, [])

                m = len(obs_rows)
                n = 2 + m
                G, h = [], []
                for k, (n_vec, rhs) in enumerate(obs_rows):
                    row = np.zeros(n)
                    row[0:2] = -n_vec
                    row[2 + k] = -1.0
                    G.append(row)
                    h.append(rhs)
                for pw, mg in wall_rows:
                    s = max(float(np.linalg.norm(pw)), self.grad_floor)
                    row = np.zeros(n)
                    row[0:2] = pw / s
                    G.append(row)
                    h.append(mg / s)

                P_diag = np.array(self.target_p_diag + [self.w_slack] * m, dtype=np.float64)
                q_vec = np.array([0.0, 0.0] + [self.w_slack_lin] * m, dtype=np.float64)
                lb = np.array([ax_lb, dl_lb] + [0.0] * m, dtype=np.float64)
                ub = np.array([ax_ub, dl_ub] + [1e6] * m, dtype=np.float64)
                
                # Apply brake clamp only if allowed
                if brake_needed and self.allow_brake_clamp:
                    ub[0] = min(ub[0], min(-self.brake_decel - cand_ax, 0.0))

                z, status = self._solve_qp(P_diag, q_vec, np.array(G), np.array(h), lb, ub)

                if z is not None:
                    du = z[:2]
                    return _finish(np.asarray(u_cand, dtype=np.float64) + du, du, status, False, z[2:])

                if top_margin < bot_margin or (top_margin == bot_margin and py > 0):
                    direction = -np.sign(p_wall[1]) if abs(p_wall[1]) > 1e-9 else -1.0
                else:
                    direction = np.sign(p_wall[1]) if abs(p_wall[1]) > 1e-9 else 1.0
                delta_fb = delta_hi if direction > 0 else delta_lo
                ax_fb = min(cand_ax, -min(p.a_max, self.brake_decel))
                u_fb = np.array([ax_fb, delta_fb], dtype=np.float64)
                return _finish(u_fb, u_fb - np.asarray(u_cand, dtype=np.float64), status, True, [])

        shield = DiagnosedShield(model_path=model_path, mode=shield_mode, r_margin=0.38, alpha1=2.5, alpha2=3.5, alpha_w=2.5, device=device)
        planner = MPPIPlanner(num_samples=500 if device == "cuda" else 100, horizon=30, v_target=6.0, r_margin=0.38, device=device)
        
        outcomes = []
        avg_speeds = []
        brake_counts = []
        no_auth_counts = []
        unreach_counts = []
        fallback_counts = []
        steps_list = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            env = ClutteredRacingEnv(num_obstacles=10, track_length=50.0, track_width=4.0, seed=seed)
            state = env.reset(start_vx=3.0)
            current_u = np.array([env.actual_ax, env.actual_delta], dtype=np.float32)
            
            done = False
            step = 0
            max_steps = 750
            dt = 0.02
            
            b_cnt = 0
            na_cnt = 0
            un_cnt = 0
            fb_cnt = 0
            speeds = []
            
            while not done and step < max_steps:
                perceived_obs = env.get_perceived_obstacles()
                obs_state = env.get_noisy_state()
                
                barrier_net = shield.barrier_net if method == 'agile_nicbf_coupled' else None
                u_nom = planner.plan(obs_state, perceived_obs, barrier_net=barrier_net)
                
                u_safe, qp_info = shield.filter_control(obs_state, current_u, u_nom, perceived_obs, dt=dt)
                
                b_cnt += int(qp_info.get("brake_override", False))
                na_cnt += int(qp_info.get("n_no_authority", 0) > 0)
                un_cnt += int(qp_info.get("n_unreachable", 0) > 0)
                fb_cnt += int(qp_info.get("wall_fallback", False))
                
                state, info = env.step(u_safe, dt=dt)
                current_u = info["actual_u"]
                step += 1
                speeds.append(float(state[3]))
                
                if info["collision"]:
                    outcomes.append("collision")
                    done = True
                elif info["out_of_bounds"]:
                    outcomes.append("out_of_bounds")
                    done = True
                elif info["success"] or state[0] >= 50.0:
                    outcomes.append("success")
                    done = True
                    
            if not done:
                outcomes.append("timeout")
                
            steps_list.append(step)
            avg_speeds.append(float(np.mean(speeds)))
            brake_counts.append(b_cnt)
            no_auth_counts.append(na_cnt)
            unreach_counts.append(un_cnt)
            fallback_counts.append(fb_cnt)
            
        success_cnt = outcomes.count("success")
        timeout_cnt = outcomes.count("timeout")
        collision_cnt = outcomes.count("collision")
        mean_spd = float(np.mean(avg_speeds))
        mean_steps = float(np.mean(steps_list))
        mean_brake = float(np.mean(brake_counts))
        mean_unreach = float(np.mean(unreach_counts))
        mean_noauth = float(np.mean(no_auth_counts))
        
        print(f"  [{method:20s}] Succ: {success_cnt}/5 | Timeouts: {timeout_cnt}/5 | Coll: {collision_cnt}/5 | Speed: {mean_spd:.2f} m/s | Steps: {mean_steps:.0f} | BrakeOverride: {mean_brake:.1f} | Unreach: {mean_unreach:.1f}")
        
        summary_results[cfg_name][method] = {
            "success": success_cnt,
            "timeout": timeout_cnt,
            "collision": collision_cnt,
            "mean_speed": round(mean_spd, 2),
            "mean_steps": round(mean_steps, 1),
            "mean_brake_overrides": round(mean_brake, 1),
            "mean_unreachable": round(mean_unreach, 1),
            "mean_no_authority": round(mean_noauth, 1),
            "outcomes": outcomes
        }

with open(os.path.join(current_dir, "diagnostic_pdiag_ablation_results.json"), "w") as f:
    json.dump(summary_results, f, indent=2)

print("\n" + "=" * 80)
print("  Diagnostics Complete & Saved to diagnostic_pdiag_ablation_results.json")
print("=" * 80)
