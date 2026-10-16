import re

with open('/home/raj/agile_nicbf/safe_controller.py', 'r') as f:
    lines = f.readlines()

# Find the start of section 3: "# 3. Wall rows"
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if "# 3. Wall rows" in line:
        start_idx = i
    if start_idx is not None and "# Closed-Loop Integration Test" in line:
        end_idx = i - 1
        break

if start_idx is None or end_idx is None:
    print("Could not find section boundaries:", start_idx, end_idx)
    exit(1)

# Find the end of filter_control before test_agile_nicbf_closed_loop
new_block = """        # 3. Dynamic Envelope Wall Containment at CG (eliminates opposing rear-bumper torque conflict)
        # Safety envelope dynamically shrinks allowed CG corridor based on vehicle yaw:
        # body_extent = half_width * |cos(psi)| + max(lf, lr) * |sin(psi)| + buffer
        wall_rows = []
        track_W = 2.0 if track_half_width is None else float(track_half_width)
        body_extent = 0.155 * abs(cos_p) + max(lf, lr) * abs(sin_p) + 0.05
        W_eff = max(1.40, track_W - body_extent)
        aw = self.alpha_w
        
        # Enforced at vehicle center of mass (l_off = 0.0) where steering authority is strictly positive
        cy_dot, cy_ddot, p_wall = self.compute_wall_hocbf_gradient(state, u_cand, l_off=0.0)
        p_wall = np.asarray(p_wall, dtype=np.float64)[:2]
        
        top_margin = -cy_ddot - 2.0 * aw * cy_dot + (aw ** 2) * (W_eff - py)
        bot_margin = cy_ddot + 2.0 * aw * cy_dot + (aw ** 2) * (py + W_eff)
        
        wall_rows.append((p_wall.copy(), float(top_margin)))
        wall_rows.append((-p_wall.copy(), float(bot_margin)))
        top_margins = [float(top_margin)]
        bot_margins = [float(bot_margin)]
        wall_margins = [float(top_margin), float(bot_margin)]
        wall_clearance = float(min(W_eff - py, py + W_eff))
        min_wall_margin = float(min(wall_margins))

        any_violated = brake_needed or min_wall_margin < 0.0 or any(rhs < 0.0 for _, rhs in obs_rows)

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
        lb = np.array([ax_lb, dl_lb] + [0.0] * m_total, dtype=np.float64)
        ub = np.array([ax_ub, dl_ub] + [1e6] * m_total, dtype=np.float64)
        
        # Apply brake clamp only if allowed
        if brake_needed and getattr(self, 'brake_clamp', True):
            ub[0] = min(ub[0], min(-self.brake_decel - cand_ax, 0.0))

        z, status = self._solve_qp(P_diag, q_vec, np.array(G), np.array(h), lb, ub)

        if z is not None:
            du = z[:2]
            return _finish(np.asarray(u_cand, dtype=np.float64) + du, du, status, False, z[2:])

        # 6. Fallback (rare numerical breakdown): smoothly steer away from nearest wall without locking brakes
        direction = -np.sign(py) if abs(py) > 0.01 else -1.0
        delta_fb = delta_hi if direction > 0 else delta_lo
        ax_fb = max(-3.0, min(cand_ax, 0.0))
        u_fb = np.array([ax_fb, delta_fb], dtype=np.float64)
        return _finish(u_fb, u_fb - np.asarray(u_cand, dtype=np.float64), status, True, [])

"""

new_lines = lines[:start_idx] + [new_block + "\n\n"] + lines[end_idx:]
with open('/home/raj/agile_nicbf/safe_controller.py', 'w') as f:
    f.writelines(new_lines)
print("Successfully replaced lines", start_idx, "to", end_idx)
