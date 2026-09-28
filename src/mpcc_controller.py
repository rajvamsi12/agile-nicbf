import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

try:
    from .dynamic_bicycle_env import DynamicBicycleModel, DynamicBicycleParams
except ImportError:
    from dynamic_bicycle_env import DynamicBicycleModel, DynamicBicycleParams

class MPCCController:
    """
    Model Predictive Contouring Control (MPCC) with Kamm's Friction Circle Limits.
    Reference: Liniger, Domahidi, Morari, 'Optimization-Based Autonomous Racing of 1:43 Scale RC Cars',
    IEEE Transactions on Control Systems Technology / Optimal Control Applications and Methods, 2015.
    
    Formulation:
    - Minimizes contouring error (e_c) and lag error (e_l) relative to track centerline
    - Maximizes progress along track (v_theta)
    - Enforces Kamm's friction circle limits: (a_x/a_max)^2 + (a_y/mu*g)^2 <= 1.0
    - Enforces obstacle avoidance repulsion half-planes
    """
    def __init__(self,
                 horizon=20,
                 dt=0.02,
                 v_target=6.0,
                 q_contour=10.0,
                 q_lag=25.0,
                 q_progress=15.0,
                 q_accel=0.1,
                 q_steer=1.0,
                 device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = DynamicBicycleModel(device=self.device)
        self.p = self.model.p
        self.N = horizon
        self.dt = dt
        self.v_target = v_target
        
        # Cost weights (Liniger et al. 2015)
        self.q_c = q_contour
        self.q_l = q_lag
        self.q_v = q_progress
        self.q_a = q_accel
        self.q_delta = q_steer
        
        # Warm-start control sequence [N, 2] -> (ax, delta)
        self.U = torch.zeros((self.N, 2), dtype=torch.float32, device=self.device)
        self.U[:, 0] = 3.0 # Initial forward throttle

    def reset(self):
        self.U = torch.zeros((self.N, 2), dtype=torch.float32, device=self.device)
        self.U[:, 0] = 3.0

    def solve(self, state, obstacles=None, max_iters=25, lr=0.05):
        """
        Solves real-time nonlinear receding-horizon MPCC optimization using fast gradient descent.
        state: [6] -> (px, py, psi, vx, vy, r)
        obstacles: [M, 3] -> (ox, oy, r_obs)
        returns: u_opt [2], info dict
        """
        t0 = time.time()
        s0 = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Initialize decision variables with warm-start
        u_seq = self.U.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([u_seq], lr=lr)
        
        obs_tensor = torch.as_tensor(obstacles, dtype=torch.float32, device=self.device) if (obstacles is not None and len(obstacles) > 0) else None
        
        for _ in range(max_iters):
            optimizer.zero_grad()
            
            # Forward simulate N steps using dynamic bicycle RK4
            states = [s0]
            curr_s = s0
            for k in range(self.N):
                u_k = torch.stack([
                    torch.clamp(u_seq[k, 0], -self.p.a_max, self.p.a_max),
                    torch.clamp(u_seq[k, 1], -self.p.delta_max, self.p.delta_max)
                ]).unsqueeze(0)
                curr_s = self.model.step_rk4_torch(curr_s, u_k, dt=self.dt)
                states.append(curr_s)
                
            traj_states = torch.cat(states[1:], dim=0) # [N, 6]
            px = traj_states[:, 0]
            py = traj_states[:, 1]
            vx = traj_states[:, 3]
            vy = traj_states[:, 4]
            r = traj_states[:, 5]
            
            # 1. Contouring and Lag Errors (Track Centerline y=0)
            e_contour = py ** 2
            cost_c = torch.sum(self.q_c * e_contour)
            
            # 2. Progress Maximization (Maximize longitudinal displacement & track speed)
            progress = px[-1] - px[0]
            cost_v = -self.q_v * progress + torch.sum(2.0 * (vx - self.v_target)**2)
            
            # 3. Track Boundary Penalty (|py| <= 1.85m)
            cost_bounds = torch.sum(torch.relu(torch.abs(py) - 1.85)**2 * 1000.0)
            
            # 4. Kamm's Friction Circle Penalty: (ax/a_max)^2 + (ay/mu*g)^2 <= 1.0
            ay = (vy * r) + (torch.diff(torch.cat([s0[:, 4:5], traj_states[:, 4:5]], dim=0), dim=0).squeeze() / self.dt)
            friction_load = (u_seq[:, 0] / self.p.a_max)**2 + (ay / (self.p.mu * self.p.g))**2
            cost_friction = torch.sum(torch.relu(friction_load - 0.95)**2 * 500.0)
            
            # 5. Obstacle Repulsion Penalty
            cost_obs = torch.tensor(0.0, device=self.device)
            if obs_tensor is not None and len(obs_tensor) > 0:
                dx = px.unsqueeze(1) - obs_tensor[:, 0].unsqueeze(0) # [N, M]
                dy = py.unsqueeze(1) - obs_tensor[:, 1].unsqueeze(0)
                r_eff = obs_tensor[:, 2].unsqueeze(0) + 0.20 # 20cm margin
                dists = torch.sqrt(torch.clamp(dx**2 + dy**2, min=1e-4))
                clearance = dists - r_eff
                cost_obs = torch.sum(torch.relu(-clearance + 0.30)**2 * 5000.0)
                
            # 6. Actuator Regularization
            cost_u = torch.sum(self.q_a * u_seq[:, 0]**2 + self.q_delta * u_seq[:, 1]**2)
            
            total_loss = cost_c + cost_v + cost_bounds + cost_friction + cost_obs + cost_u
            total_loss.backward()
            optimizer.step()
            
        # Shift control horizon for warm start
        self.U = torch.cat([u_seq[1:].detach(), u_seq[-1:].detach()], dim=0)
        
        u_opt = np.array([
            float(np.clip(u_seq[0, 0].item(), -self.p.a_max, self.p.a_max)),
            float(np.clip(u_seq[0, 1].item(), -self.p.delta_max, self.p.delta_max))
        ], dtype=np.float32)
        
        latency = (time.time() - t0) * 1000.0
        return u_opt, {"latency_ms": latency, "loss": float(total_loss.item())}
