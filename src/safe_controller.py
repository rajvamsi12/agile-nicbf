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
    Microsecond Neural Integral Control Barrier Function (NICBF) Safety Shield.
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
                 gamma_u=2.0,         # Normalized actuator input-barrier class-K gain
                 r_margin=0.26,       # Safety boundary margin (meters, ensures clearance outside 0.20m crash boundary)
                 w_slack=1000.0,      # Feasibility relaxation weight
                 solver='quadprog',   # Exact active-set solver
                 device='cpu'):
        
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = DynamicBicycleModel(device=self.device)
        self.p = self.model.p
        
        self.mode = mode
        self.alpha_nicbf = alpha_nicbf
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha_w = alpha_w
        self.gamma_u = gamma_u
        self.r_margin = r_margin
        self.w_slack = w_slack
        self.solver = solver
        
        # Load trained Neural Barrier Network
        self.barrier_net = NeuralBarrierNet(input_dim=6, hidden_dim=256).to(self.device)
        
        # Robust model path resolution
        resolved_path = model_path
        if resolved_path is None or not os.path.exists(resolved_path):
            candidates = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "neural_icbf_6d.pth"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "neural_icbf_6d.pth"),
                "/home/raj/agile_nicbf/models/neural_icbf_6d.pth",
                os.path.abspath("models/neural_icbf_6d.pth"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    resolved_path = cand
                    break
                    
        if resolved_path is not None and os.path.exists(resolved_path):
            self.barrier_net.load_state_dict(torch.load(resolved_path, map_location=self.device, weights_only=False))
            self.barrier_net.eval()
            print(f"[Stage 4] Loaded Neural Barrier Network from {resolved_path} (mode={self.mode})")
        else:
            print(f"[Stage 4] Warning: Model checkpoint not found. Running with uninitialized weights.")

    def compute_neural_icbf_gradient(self, state, u_cand, obs):
        """
        Computes exact Lie derivatives and control Jacobian using the trained 6D Neural Barrier Network.
        Relative degree = 1 with respect to control u = [ax, delta] because h_theta depends on (vx, vy, r).
        Condition: h_dot + alpha_nicbf * h >= 0
        """
        px, py, psi, vx, vy, r = state
        ox, oy, ro = obs
        dx = px - ox
        dy = py - oy
        r_eff = ro + self.r_margin
        scale = 0.60 / r_eff  # Normalize effective obstacle boundary to 0.60m nominal training radius
        
        # 6D relative state tensor for Neural Barrier Network
        x_rel = torch.tensor([[dx * scale, dy * scale, psi, vx, vy, r]], dtype=torch.float32, requires_grad=True, device=self.device)
        h_val = self.barrier_net(x_rel).squeeze()
        grad_x = torch.autograd.grad(h_val, x_rel, create_graph=True)[0].squeeze()
        
        u_t = torch.as_tensor(u_cand, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_()
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        f = self.model.continuous_dynamics_torch(s_t, u_t).squeeze()
        f_effective = torch.stack([f[0] * scale, f[1] * scale, f[2], f[3], f[4], f[5]])
        h_dot = torch.sum(grad_x * f_effective)
        
        nicbf_margin = h_dot + self.alpha_nicbf * h_val
        p_obs = torch.autograd.grad(h_dot, u_t, create_graph=False)[0].squeeze().detach().cpu().numpy().astype(np.float64)
        return h_val.item(), nicbf_margin.item(), p_obs

    def compute_obstacle_hocbf_gradient(self, state, u_cand, obs):
        """
        Computes exact 2nd-order Lie derivatives and control Jacobian for analytical geometric obstacle avoidance.
        Relative degree = 2: position -> velocity -> acceleration -> control u.
        Condition: h_ddot + (alpha1 + alpha2)*h_dot + alpha1*alpha2*h >= 0
        """
        px, py, psi, vx, vy, r = state
        ox, oy, ro = obs
        dx = px - ox
        dy = py - oy
        r_eff = ro + self.r_margin
        
        h = dx**2 + dy**2 - r_eff**2
        px_dot = vx * np.cos(psi) - vy * np.sin(psi)
        py_dot = vx * np.sin(psi) + vy * np.cos(psi)
        h_dot = 2.0 * dx * px_dot + 2.0 * dy * py_dot
        
        u_t = torch.as_tensor(u_cand, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_()
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        f = self.model.continuous_dynamics_torch(s_t, u_t).squeeze()
        
        px_ddot = (f[3] - vy * r) * torch.cos(s_t[0, 2]) - (f[4] + vx * r) * torch.sin(s_t[0, 2])
        py_ddot = (f[3] - vy * r) * torch.sin(s_t[0, 2]) + (f[4] + vx * r) * torch.cos(s_t[0, 2])
        
        h_ddot = 2.0 * (px_dot**2 + py_dot**2) + 2.0 * dx * px_ddot + 2.0 * dy * py_ddot
        hocbf_margin = h_ddot + (self.alpha1 + self.alpha2) * h_dot + (self.alpha1 * self.alpha2) * h
        p_obs = torch.autograd.grad(h_ddot, u_t, create_graph=False)[0].squeeze().detach().cpu().numpy().astype(np.float64)
        return h, hocbf_margin.item(), p_obs

    def compute_wall_hocbf_gradient(self, state, current_u):
        """
        Computes exact 2nd-order Lie derivatives and control Jacobian for track containment.
        Relative degree = 2: py -> py_dot -> py_ddot(u).
        """
        px, py, psi, vx, vy, r = state
        u_tensor = torch.as_tensor(current_u, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_()
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        f = self.model.continuous_dynamics_torch(state_tensor, u_tensor).squeeze()
        py_dot = vx * np.sin(psi) + vy * np.cos(psi)
        py_ddot = (f[3] - vy * r) * torch.sin(state_tensor[0, 2]) + (f[4] + vx * r) * torch.cos(state_tensor[0, 2])
        
        p_wall = torch.autograd.grad(py_ddot, u_tensor, create_graph=False)[0].squeeze().detach().cpu().numpy().astype(np.float64)
        return py_dot, py_ddot.item(), p_wall

    def filter_control(self, state, current_u, u_nom, obstacles, dt=0.02, track_half_width=1.85):
        """
        Filters nominal control u_nom using the online multi-obstacle QP safety shield.
        state: [6] -> (px, py, psi, vx, vy, r)
        current_u: [2] -> (ax, delta)
        u_nom: [2] -> (ax_nom, delta_nom) from MPPI
        obstacles: [M, 3] -> (ox, oy, r_obs)
        returns: safe_u [2], qp_info dict
        """
        t0 = time.time()
        px, py, psi, vx, vy, r = state
        ax = float(np.clip(u_nom[0], -self.p.a_max, self.p.a_max))
        delta = float(np.clip(u_nom[1], -self.p.delta_max, self.p.delta_max))
        u_cand = np.array([ax, delta], dtype=np.float32)
        
        G_rows = []
        h_vals_list = []
        min_h_nom = 100.0
        
        cos_p = np.cos(psi)
        sin_p = np.sin(psi)
        
        # 1. Multi-Obstacle Forward Hazard Check
        for obs in obstacles:
            ox, oy, r_obs = obs
            dx_body = (ox - px) * cos_p + (oy - py) * sin_p
            dy_body = -(ox - px) * sin_p + (oy - py) * cos_p
            r_eff = r_obs + self.r_margin
            
            # Active collision risk: within forward/nearby hazard window
            if -r_eff <= dx_body <= 6.0 and abs(dy_body) < (r_eff + 0.60):
                if self.mode == 'neural_icbf':
                    h_val, margin, p_obs = self.compute_neural_icbf_gradient(state, u_cand, obs)
                else:
                    h_val, margin, p_obs = self.compute_obstacle_hocbf_gradient(state, u_cand, obs)
                    
                if h_val < min_h_nom:
                    min_h_nom = h_val
                    
                if margin < 0.0:  # Candidate control violates safety invariance!
                    # p_obs^T du >= -margin <=> -p_obs^T du <= margin
                    G_rows.append([-p_obs[0], -p_obs[1], -1.0])
                    h_vals_list.append(float(margin))
                    
                # Formal Longitudinal Approach HOCBF (Relative Degree 2 along body x-axis):
                # When obstacle is directly ahead in the vehicle's swept path, enforce longitudinal stopping barrier
                # h_fwd = dx_body - r_eff >= 0 => ddot(h_fwd) + (alpha1 + alpha2)*dot(h_fwd) + alpha1*alpha2*h_fwd >= 0
                # dot(dx_body) = -vx + dy_body * r, ddot(dx_body) ~= -ax
                # Condition: -ax - (alpha1 + alpha2)*(vx - dy_body * r) + alpha1*alpha2*(dx_body - r_eff) >= -eps
                if 0.0 <= dx_body <= 2.0 and abs(dy_body) <= r_eff:
                    fwd_margin = -ax - (self.alpha1 + self.alpha2) * (vx - dy_body * r) + (self.alpha1 * self.alpha2) * (dx_body - r_eff)
                    if dx_body <= (r_eff + 0.15) and vx < 0.5:
                        fwd_margin = min(fwd_margin, -ax)
                    if fwd_margin < 0.0:
                        G_rows.append([1.0, 0.0, 0.0])
                        h_vals_list.append(float(fwd_margin))
                    
        # 2. Track Boundary Wall Higher-Order CBF (Relative Degree 2)
        py_dot, py_ddot, p_wall = self.compute_wall_hocbf_gradient(state, u_cand)
        
        # Top Wall: h_top = W - py >= 0  =>  py_ddot <= -2*alpha*py_dot + alpha^2*(W - py)
        w_top_margin = -py_ddot - 2.0 * self.alpha_w * py_dot + (self.alpha_w**2) * (track_half_width - py)
        if w_top_margin < 0.0:
            G_rows.append([p_wall[0], p_wall[1], -1.0])
            h_vals_list.append(float(w_top_margin))
            
        # Bottom Wall: h_bot = py + W >= 0  =>  py_ddot >= -2*alpha*py_dot - alpha^2*(py + W)
        w_bot_margin = py_ddot + 2.0 * self.alpha_w * py_dot + (self.alpha_w**2) * (py + track_half_width)
        if w_bot_margin < 0.0:
            G_rows.append([-p_wall[0], -p_wall[1], -1.0])
            h_vals_list.append(float(w_bot_margin))
            
        # Compute overall physical safety clearance (considering both obstacle and wall proximity)
        min_wall_clearance = float(min(track_half_width - py, py + track_half_width))
        overall_margin = min(min_h_nom, min_wall_clearance) if min_h_nom < 90.0 else min_wall_clearance
        
        # 3. If all constraints satisfied under candidate control -> zero intervention!
        if len(G_rows) == 0:
            u_safe = np.copy(u_cand)
            u_safe[0] = np.clip(u_safe[0], -self.p.a_max, self.p.a_max)
            u_safe[1] = np.clip(u_safe[1], -self.p.delta_max, self.p.delta_max)
            return u_safe, {"qp_active": False, "v_star": np.zeros(2), "min_h_val": overall_margin, "hu": 1.0, "latency_ms": (time.time()-t0)*1000.0}
            
        # 4. Normalized actuator input barrier, following the ICBF input-bound construction
        # h_u = 1 - ((a_x/a_max)^2 + (delta/delta_max)^2) >= 0.
        # Tire-force saturation is handled in the dynamics model, not by this QP row.
        hu = 1.0 - ((ax / self.p.a_max)**2 + (delta / self.p.delta_max)**2)
        pu = np.array([-2.0 * ax / (self.p.a_max**2), -2.0 * delta / (self.p.delta_max**2)], dtype=np.float64)
        rhs_u = float(self.gamma_u * hu)
        G_rows.append([-pu[0], -pu[1], -1.0])
        h_vals_list.append(rhs_u)
        
        # Non-negative slack: -eps <= 0
        G_rows.append([0.0, 0.0, -1.0])
        h_vals_list.append(0.0)
        
        # 5. Assemble and Solve Minimal-Intervention QP: min ||(u - u_cand) / u_max||^2 + w_slack * eps^2
        P_mat = np.diag([1.0 / (self.p.a_max**2), 1.0 / (self.p.delta_max**2), self.w_slack])
        q_vec = np.zeros(3, dtype=np.float64)
        G_mat = np.array(G_rows, dtype=np.float64)
        h_vec = np.array(h_vals_list, dtype=np.float64)
        
        lb = np.array([-self.p.a_max - ax, -self.p.delta_max - delta, 0.0], dtype=np.float64)
        ub = np.array([self.p.a_max - ax, self.p.delta_max - delta, 1e6], dtype=np.float64)
        
        du = np.zeros(2, dtype=np.float64)
        qp_active = False
        
        try:
            z_opt = solve_qp(P_mat, q_vec, G_mat, h_vec, lb=lb, ub=ub, solver=self.solver)
            if z_opt is not None:
                du = z_opt[:2]
                qp_active = np.linalg.norm(du) > 1e-3
            else:
                # Emergency safe braking fallback when QP is infeasible
                du = np.array([-self.p.a_max - ax, -delta], dtype=np.float64)
                qp_active = True
        except Exception:
            # Emergency safe braking fallback
            du = np.array([-self.p.a_max - ax, -delta], dtype=np.float64)
            qp_active = True
            
        u_safe = u_cand + du
        u_safe[0] = np.clip(u_safe[0], -self.p.a_max, self.p.a_max)
        u_safe[1] = np.clip(u_safe[1], -self.p.delta_max, self.p.delta_max)
        
        latency = (time.time() - t0) * 1000.0
        info = {
            "qp_active": qp_active,
            "v_star": du,
            "min_h_val": overall_margin,
            "hu": hu,
            "latency_ms": latency
        }
        return u_safe, info


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
    planner = MPPIPlanner(num_samples=3000 if torch.cuda.is_available() else 500, horizon=35, v_target=6.0, device=device)
    shield = NeuralICBFController(model_path=model_path, mode='neural_icbf', r_margin=0.26, alpha1=2.5, alpha2=3.5, alpha_w=3.5, device=device)
    
    current_u = np.array([2.0, 0.0])
    total_steps = 0
    start_time = time.time()
    done = False
    
    qp_interventions = 0
    total_latency = []
    
    while not done and total_steps < 600:
        px, py = state[0], state[1]
        nearby_obs = []
        for ox, oy, r_obs in env.obstacles:
            if -2.0 <= (ox - px) <= 15.0 and abs(oy - py) <= 3.0:
                nearby_obs.append([ox, oy, r_obs])
        nearby_obs = np.array(nearby_obs, dtype=np.float32) if len(nearby_obs) > 0 else np.zeros((0, 3), dtype=np.float32)
        
        t0 = time.time()
        # 1. MPPI Nominal Plan
        u_nom = planner.plan(state, nearby_obs)
        # 2. Neural ICBF Multi-Obstacle QP Shield
        u_safe, qp_info = shield.filter_control(state, current_u, u_nom, nearby_obs, dt=0.02)
        step_latency = (time.time() - t0) * 1000.0
        total_latency.append(step_latency)
        
        if qp_info["qp_active"]:
            qp_interventions += 1
            
        current_u = u_safe
        state, info = env.step(u_safe, dt=0.02)
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
