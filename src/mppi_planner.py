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
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = torch.device(device)
        self.model = model if model is not None else DynamicBicycleModel(device=self.device)
        self.p = self.model.p
        
        self.horizon = horizon
        self.K = num_samples
        self.dt = dt
        self.lam = lambda_param
        self.v_target = v_target
        
        # Noise covariance
        self.noise_std = torch.tensor([sigma_accel, sigma_steer], device=self.device)
        
        # Nominal control sequence: [H, 2] -> (ax, delta)
        self.U = torch.zeros((self.horizon, 2), device=self.device)
        self.U[:, 0] = 3.0 # Initial forward throttle guess

    def reset(self):
        self.U = torch.zeros((self.horizon, 2), device=self.device)
        self.U[:, 0] = 3.0

    def compute_trajectory_costs(self, states, controls, noise, obstacles_tensor):
        """
        Calculates the cost of each of the K candidate trajectories in parallel on GPU.
        states: [K, H, 6] -> (px, py, psi, vx, vy, r)
        controls: [K, H, 2] -> (ax, delta)
        obstacles_tensor: [M, 3] -> (ox, oy, r_obs)
        """
        # 1. Forward Progress Incentive (Fix for Bug 4: Anti-Stall)
        px = states[:, :, 0]
        progress = px[:, -1] - px[:, 0]
        cost_progress = -progress * 25.0 # High reward for moving down the track
        
        # 2. Target Speed Tracking Cost
        vx = states[:, :, 3]
        cost_speed = torch.sum((vx - self.v_target) ** 2, dim=1) * 2.0
        
        # 3. Track Centerline Progress & Boundary Penalty
        py = states[:, :, 1]
        cost_centerline = torch.sum(py ** 2, dim=1) * 0.5
        
        # Steep penalty if vehicle leaves track boundaries (|py| > 1.8m)
        cost_bounds = torch.sum(torch.relu(torch.abs(py) - 1.8) * 2000.0, dim=1)
        
        # 4. Obstacle Collision & Proximity Cost
        cost_obstacles = torch.zeros(self.K, device=self.device)
        if obstacles_tensor is not None and len(obstacles_tensor) > 0:
            px_exp = states[:, :, 0].unsqueeze(2) # [K, H, 1]
            py_exp = states[:, :, 1].unsqueeze(2) # [K, H, 1]
            
            ox = obstacles_tensor[:, 0].unsqueeze(0).unsqueeze(0) # [1, 1, M]
            oy = obstacles_tensor[:, 1].unsqueeze(0).unsqueeze(0) # [1, 1, M]
            r_obs = obstacles_tensor[:, 2].unsqueeze(0).unsqueeze(0) # [1, 1, M]
            
            dist_sq = (px_exp - ox) ** 2 + (py_exp - oy) ** 2
            dists = torch.sqrt(torch.clamp(dist_sq, min=1e-4))
            
            # Distance margin to obstacle boundary (car radius = 0.20m)
            margin = dists - (r_obs + 0.20)
            
            # Exponential barrier penalty for nearby obstacles (smooth gradient away from obstacles)
            obs_penalties = torch.exp(-4.0 * torch.clamp(margin, min=0.0)) * torch.clamp(1.5 - margin, min=0.0)
            
            # Hard crash penalty if trajectory breaches safety margin
            crash_mask = (margin <= 0.0).any(dim=2).any(dim=1) # [K]
            cost_obstacles = torch.sum(obs_penalties, dim=(1, 2)) * 15.0 + crash_mask.float() * 100000.0
            
        # 5. Control Effort & Smoothness Regularization
        cost_control = torch.sum(controls ** 2, dim=(1, 2)) * 0.05
        
        total_costs = cost_progress + cost_speed + cost_centerline + cost_bounds + cost_obstacles + cost_control
        return total_costs

    def plan(self, current_state, obstacles):
        """
        Runs one iteration of MPPI optimization using RK4 integration.
        current_state: numpy array [6] -> (px, py, psi, vx, vy, r)
        obstacles: numpy array [M, 3] -> (ox, oy, r_obs)
        returns: optimal immediate control [ax, delta]
        """
        state_0 = torch.tensor(current_state, dtype=torch.float32, device=self.device).repeat(self.K, 1)
        obstacles_tensor = torch.tensor(obstacles, dtype=torch.float32, device=self.device) if len(obstacles) > 0 else None
        
        # 1. Sample control noise perturbations epsilon: [K, H, 2]
        noise = torch.randn((self.K, self.horizon, 2), device=self.device) * self.noise_std
        
        candidate_controls = self.U.unsqueeze(0) + noise
        candidate_controls[:, :, 0] = torch.clamp(candidate_controls[:, :, 0], -self.p.a_max, self.p.a_max)
        candidate_controls[:, :, 1] = torch.clamp(candidate_controls[:, :, 1], -self.p.delta_max, self.p.delta_max)
        
        # 2. Rollout dynamic trajectories on GPU using RK4 (Fix for Bug 3)
        states_rollout = torch.zeros((self.K, self.horizon, 6), device=self.device)
        curr_x = state_0.clone()
        
        for t in range(self.horizon):
            u_t = candidate_controls[:, t, :]
            curr_x = self.model.step_rk4_torch(curr_x, u_t, dt=self.dt)
            states_rollout[:, t, :] = curr_x
            
        # 3. Compute trajectory costs
        costs = self.compute_trajectory_costs(states_rollout, candidate_controls, noise, obstacles_tensor)
        
        # 4. Compute importance weights
        min_cost = torch.min(costs)
        weights = torch.exp(-(costs - min_cost) / self.lam)
        weights = weights / (torch.sum(weights) + 1e-8) # [K]
        
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
