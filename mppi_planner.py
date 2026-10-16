import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

# Clean import handling
try:
    from .dynamic_bicycle_env import DynamicBicycleModel, DynamicBicycleParams, ClutteredRacingEnv
except ImportError:
    from dynamic_bicycle_env import DynamicBicycleModel, DynamicBicycleParams, ClutteredRacingEnv

class MPPIPlanner:
    """
    GPU-Accelerated Model Predictive Path Integral (MPPI) Planner.
    Reference: Williams et al., "Information-Theoretic Model Predictive Control: Theory 
    and Applications to Autonomous Flight and Driving," IEEE Transactions on Robotics, 34(6):1603-1622, 2018.
    
    Generates high-speed agile nominal trajectory u_nom through cluttered obstacle fields.
    """
    def __init__(self,
                 model=None,
                 horizon=35,          # 35 steps * 0.02s = 0.7s lookahead (agile slalom lookahead)
                 num_samples=3000,    # K parallel rollouts on GPU
                 dt=0.02,             # 50 Hz control frequency
                 lambda_param=0.5,    # MPPI temperature parameter
                 sigma_steer=0.18,    # Steering noise std dev (rad)
                 sigma_accel=2.0,     # Acceleration noise std dev (m/s^2)
                 v_target=6.0,        # Target longitudinal speed (m/s)
                 r_margin=0.38,       # Obstacle safety buffer (m)
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = torch.device(device)
        self.model = model if model is not None else DynamicBicycleModel(device=self.device)
        self.p = self.model.p
        
        self.horizon = horizon
        self.K = num_samples
        self.dt = dt
        self.lam = lambda_param
        self.v_target = v_target
        self.r_margin = r_margin
        self.n_eff = float(self.K)
        
        # Noise covariance
        self.noise_std = torch.tensor([sigma_accel, sigma_steer], device=self.device)
        
        # Nominal control sequence: [H, 2] -> (ax, delta)
        self.U = torch.zeros((self.horizon, 2), device=self.device)
        self.U[:, 0] = 3.0 # Initial forward throttle guess

    def reset(self):
        self.U = torch.zeros((self.horizon, 2), device=self.device)
        self.U[:, 0] = 3.0
        self.n_eff = float(self.K)

    def compute_trajectory_costs(self, states, controls, noise, obstacles_tensor, barrier_net=None):
        """
        Calculates the cost of each of the K candidate trajectories in parallel on GPU.
        states: [K, H, 6] -> (px, py, psi, vx, vy, r)
        controls: [K, H, 2] -> (ax, delta)
        obstacles_tensor: [M, 3] -> (ox, oy, r_obs)
        """
        # 1. Forward Progress Incentive (High reward for forward velocity)
        px = states[:, :, 0]
        progress = px[:, -1] - px[:, 0]
        cost_progress = -progress * 40.0
        
        # 2. Target Speed Tracking Cost
        vx = states[:, :, 3]
        cost_speed = torch.sum((vx - self.v_target) ** 2, dim=1) * 2.0
        
        # 3. Track Centerline Progress & Boundary Penalty
        py = states[:, :, 1]
        cost_centerline = torch.sum(py ** 2, dim=1) * 0.5
        
        # Steep penalty if vehicle leaves track boundaries (|py| > 1.75m)
        # Note: 0.25m clearance from 2.0m wall covers lateral extent up to ~20 deg yaw
        cost_bounds = torch.sum(torch.relu(torch.abs(py) - 1.58) * 5000.0, dim=1)
        
        # 3b. Track Heading Alignment Cost (penalizes spinning or turning backwards)
        cost_heading = torch.sum(states[:, :, 2] ** 2, dim=1) * 3.0
        
        # 4. Obstacle Collision & Proximity Cost (3-Disk Footprint)
        cost_obstacles = torch.zeros(self.K, device=self.device)
        cost_barrier = torch.zeros(self.K, device=self.device)
        
        if obstacles_tensor is not None and len(obstacles_tensor) > 0:
            px = states[:, :, 0]
            py = states[:, :, 1]
            psi = states[:, :, 2]
            cos_psi = torch.cos(psi)
            sin_psi = torch.sin(psi)
            
            # 3 disk centers along centerline: rear axle (-lr), CG (0), front axle (+lf)
            # Disk radius r_disk = 0.21m geometrically covers rectangular body corners (0.58m x 0.31m)
            r_disk = 0.21
            x_rear = px - self.p.lr * cos_psi
            y_rear = py - self.p.lr * sin_psi
            x_cg = px
            y_cg = py
            x_front = px + self.p.lf * cos_psi
            y_front = py + self.p.lf * sin_psi
            
            # Shape: [K, H, 3, 1]
            x_disks = torch.stack([x_rear, x_cg, x_front], dim=2).unsqueeze(3)
            y_disks = torch.stack([y_rear, y_cg, y_front], dim=2).unsqueeze(3)
            
            # Obstacles: [1, 1, 1, M]
            ox = obstacles_tensor[:, 0].view(1, 1, 1, -1)
            oy = obstacles_tensor[:, 1].view(1, 1, 1, -1)
            r_obs = obstacles_tensor[:, 2].view(1, 1, 1, -1)
            
            dist_sq = (x_disks - ox) ** 2 + (y_disks - oy) ** 2
            dists = torch.sqrt(torch.clamp(dist_sq, min=1e-4)) # [K, H, 3, M]
            
            # Distance margin to obstacle boundary for each disk
            margin = dists - (r_obs + r_disk)
            
            # Tight exponential barrier penalty for nearby obstacles (allows navigating tight gaps)
            obs_penalties = torch.exp(-5.0 * torch.clamp(margin, min=0.0)) * torch.clamp(0.80 - margin, min=0.0)
            
            # Hard crash penalty if trajectory breaches safety margin on any disk
            crash_mask = (margin <= 0.0).any(dim=3).any(dim=2).any(dim=1) # [K]
            cost_obstacles = torch.sum(obs_penalties, dim=(1, 2, 3)) * 25.0 + crash_mask.float() * 100000.0
            
            # 4b. Neural ICBF Invariant Loss on GPU (Coupled CBF-MPPI)
            if barrier_net is not None:
                # Window closest obstacle to current rollout (evaluating at CG)
                dist_cg_sq = (px.unsqueeze(2) - obstacles_tensor[:, 0].view(1, 1, -1)) ** 2 + \
                             (py.unsqueeze(2) - obstacles_tensor[:, 1].view(1, 1, -1)) ** 2
                dists_cg = torch.sqrt(torch.clamp(dist_cg_sq, min=1e-4)) # [K, H, M]
                min_dist_idx = torch.argmin(dists_cg.mean(dim=1), dim=1) # [K]
                obs_selected = obstacles_tensor[min_dist_idx] # [K, 3]
                
                dx = (states[:, :, 0] - obs_selected[:, 0].unsqueeze(1))
                dy = (states[:, :, 1] - obs_selected[:, 1].unsqueeze(1))
                r_eff = obs_selected[:, 2].unsqueeze(1) + self.r_margin
                scale = 0.60 / r_eff
                
                # Flatten [K*H, 6]
                psi_flat = states[:, :, 2].reshape(-1, 1)
                vx_flat = states[:, :, 3].reshape(-1, 1)
                vy_flat = states[:, :, 4].reshape(-1, 1)
                r_flat = states[:, :, 5].reshape(-1, 1)
                dx_flat = (dx * scale).reshape(-1, 1)
                dy_flat = (dy * scale).reshape(-1, 1)
                
                x_rel_all = torch.cat([dx_flat, dy_flat, psi_flat, vx_flat, vy_flat, r_flat], dim=1)
                with torch.no_grad():
                    h_vals = barrier_net(x_rel_all).view(self.K, self.horizon)
                
                # Penalize safety barrier violation
                barrier_violations = torch.relu(-h_vals)
                cost_barrier = torch.sum(barrier_violations, dim=1) * 100.0
            
        # 5. Control Effort & Smoothness Regularization
        cost_control = torch.sum(controls ** 2, dim=(1, 2)) * 0.05
        
        total_costs = cost_progress + cost_speed + cost_centerline + cost_bounds + cost_heading + cost_obstacles + cost_barrier + cost_control
        return total_costs

    def plan(self, current_state, obstacles, barrier_net=None):
        """
        Runs one iteration of MPPI optimization using RK4 integration.
        current_state: numpy array [6] -> (px, py, psi, vx, vy, r)
        obstacles: numpy array [M, 3] -> (ox, oy, r_obs)
        barrier_net: optional trained NeuralBarrierNet for coupled CBF-MPPI planning
        returns: optimal immediate control [ax, delta]
        """
        state_0 = torch.tensor(current_state, dtype=torch.float32, device=self.device).repeat(self.K, 1)
        obstacles_tensor = torch.tensor(obstacles, dtype=torch.float32, device=self.device) if len(obstacles) > 0 else None
        
        # 1. Sample control noise perturbations epsilon: [K, H, 2]
        noise = torch.randn((self.K, self.horizon, 2), device=self.device) * self.noise_std
        
        candidate_controls = self.U.unsqueeze(0) + noise
        candidate_controls[:, :, 0] = torch.clamp(candidate_controls[:, :, 0], -self.p.a_max, self.p.a_max)
        candidate_controls[:, :, 1] = torch.clamp(candidate_controls[:, :, 1], -self.p.delta_max, self.p.delta_max)
        
        # 2. Rollout dynamic trajectories on GPU using RK4
        states_rollout = torch.zeros((self.K, self.horizon, 6), device=self.device)
        curr_x = state_0.clone()
        
        for t in range(self.horizon):
            u_t = candidate_controls[:, t, :]
            curr_x = self.model.step_rk4_torch(curr_x, u_t, dt=self.dt)
            states_rollout[:, t, :] = curr_x
            
        # 3. Compute trajectory costs
        costs = self.compute_trajectory_costs(states_rollout, candidate_controls, noise, obstacles_tensor, barrier_net=barrier_net)
        # Exposed for downstream predictive safety filters (no effect on planning)
        self.last_samples = candidate_controls.detach()
        self.last_costs = costs.detach()
        
        # 4. Compute importance weights without cost washout
        # Rollouts with crash penalty (cost >= 50,000) are excluded from softmax to prevent washing out
        # legitimate non-crash cost differences (progress, tracking).
        non_crash_mask = costs < 50000.0
        n_surv = int(torch.sum(non_crash_mask).item())
        
        if n_surv >= 5:
            self.last_n_surv = n_surv
            valid_costs = costs[non_crash_mask]
            min_cost = torch.min(valid_costs)
            shifted_costs = valid_costs - min_cost
            
            # Target effective sample size: ~10% of K, capped by available surviving rollouts
            target_neff = min(float(n_surv) * 0.8, max(5.0, 0.1 * self.K))
            
            # Fast bisection on lambda to achieve target N_eff
            lam_low = 1e-3
            lam_high = 100.0 * (torch.std(shifted_costs).item() + 1.0)
            
            for _ in range(10):
                lam_mid = 0.5 * (lam_low + lam_high)
                w_mid = torch.exp(-shifted_costs / lam_mid)
                w_mid = w_mid / (torch.sum(w_mid) + 1e-8)
                n_eff_mid = (1.0 / (torch.sum(w_mid ** 2) + 1e-8)).item()
                if n_eff_mid < target_neff:
                    lam_low = lam_mid
                else:
                    lam_high = lam_mid
            
            best_lam = 0.5 * (lam_low + lam_high)
            weights_valid = torch.exp(-shifted_costs / best_lam)
            weights_valid = weights_valid / (torch.sum(weights_valid) + 1e-8)
            
            weights = torch.zeros(self.K, device=self.device)
            weights[non_crash_mask] = weights_valid
            self.n_eff = (1.0 / (torch.sum(weights_valid ** 2) + 1e-8)).item()
        else:
            self.last_n_surv = n_surv
            # Fallback if almost all rollouts crash: standard softmax with base lambda
            min_cost = torch.min(costs)
            weights = torch.exp(-(costs - min_cost) / self.lam)
            weights = weights / (torch.sum(weights) + 1e-8)
            self.n_eff = (1.0 / (torch.sum(weights ** 2) + 1e-8)).item()
        
        # 5. Weighted average update of the control sequence
        weighted_noise = torch.sum(weights.unsqueeze(1).unsqueeze(2) * noise, dim=0) # [H, 2]
        self.U = self.U + weighted_noise
        
        self.U[:, 0] = torch.clamp(self.U[:, 0], -self.p.a_max, self.p.a_max)
        self.U[:, 1] = torch.clamp(self.U[:, 1], -self.p.delta_max, self.p.delta_max)
        
        u_opt = self.U[0, :].detach().cpu().numpy()
        
        # Shift horizon forward by 1 step
        self.U[:-1, :] = self.U[1:, :].clone()
        self.U[-1, :] = self.U[-2, :].clone()
        
        return u_opt


# ---------------------------------------------------------------------------
# Test & Sanity Check Simulation
# ---------------------------------------------------------------------------
def test_mppi_in_cluttered_arena():
    print(f"\n[Stage 3] Testing MPPI Swerve Planner with RK4 and Anti-Stall Progress Cost...")
    env = ClutteredRacingEnv(num_obstacles=20, track_length=50.0, seed=42)
    state = env.reset(start_vx=3.0)
    
    planner = MPPIPlanner(num_samples=3000, horizon=35, v_target=6.0)
    
    total_steps = 0
    start_time = time.time()
    done = False
    trajectory = []
    
    while not done and total_steps < 600:
        px, py = state[0], state[1]
        nearby_obs = []
        for ox, oy, r_obs in env.obstacles:
            if -2.0 <= (ox - px) <= 15.0 and abs(oy - py) <= 3.0:
                nearby_obs.append([ox, oy, r_obs])
        nearby_obs = np.array(nearby_obs, dtype=np.float32) if len(nearby_obs) > 0 else np.zeros((0, 3), dtype=np.float32)
        
        t0 = time.time()
        u_nom = planner.plan(state, nearby_obs)
        step_latency = (time.time() - t0) * 1000.0
        
        state, info = env.step(u_nom, dt=0.02)
        trajectory.append((state[0], state[1], state[3], step_latency))
        total_steps += 1
        
        if info["collision"]:
            print(f"Step {total_steps:03d} | Collided with obstacle at x={state[0]:.2f}m, y={state[1]:.2f}m! (Sampling noise limitation)")
            done = True
        elif info["out_of_bounds"]:
            print(f"Step {total_steps:03d} | Out of bounds at y={state[1]:.2f}m!")
            done = True
        elif info["success"]:
            print(f"Step {total_steps:03d} | SUCCESS: Crossed finish line at x={state[0]:.2f}m!")
            done = True
            
    elapsed = time.time() - start_time
    avg_latency = np.mean([t[3] for t in trajectory])
    max_speed = np.max([t[2] for t in trajectory])
    print(f"[Stage 3] Finished: {total_steps} steps in {elapsed:.2f}s | Avg MPPI Solve Latency: {avg_latency:.2f} ms | Max Speed: {max_speed:.2f} m/s")

if __name__ == "__main__":
    test_mppi_in_cluttered_arena()
