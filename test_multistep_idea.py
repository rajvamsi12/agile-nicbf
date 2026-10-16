import sys, os, time
sys.path.insert(0, '/home/raj/agile_nicbf')
sys.path.insert(0, '/home/raj/agile_nicbf/src')

import numpy as np
import torch
from dynamic_bicycle_env import ClutteredRacingEnv
from mppi_planner import MPPIPlanner
from safe_controller import NeuralICBFController
from stress_grid import find_model

class MultiStepNICBFPlanner(MPPIPlanner):
    def compute_trajectory_costs(self, states, controls, noise, obstacles_tensor, barrier_net=None):
        # 1. Forward progress
        px = states[:, :, 0]
        progress = px[:, -1] - px[:, 0]
        cost_progress = -progress * 40.0
        
        # 2. Target speed tracking
        vx = states[:, :, 3]
        cost_speed = torch.sum((vx - self.v_target) ** 2, dim=1) * 2.0
        
        # 3. Centerline progress
        py = states[:, :, 1]
        cost_centerline = torch.sum(py ** 2, dim=1) * 0.5
        
        # 3b. Heading alignment
        cost_heading = torch.sum(states[:, :, 2] ** 2, dim=1) * 3.0
        
        # 4. Geometry and Boundary
        psi = states[:, :, 2]
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)
        
        r_disk = 0.21
        x_rear = px - self.p.lr * cos_psi
        y_rear = py - self.p.lr * sin_psi
        x_cg = px
        y_cg = py
        x_front = px + self.p.lf * cos_psi
        y_front = py + self.p.lf * sin_psi
        
        x_disks = torch.stack([x_rear, x_cg, x_front], dim=2).unsqueeze(3)
        y_disks = torch.stack([y_rear, y_cg, y_front], dim=2).unsqueeze(3)
        
        # Wall breach: track half-width = 2.0m. Disk edge crosses wall if |y_disk| >= 2.0 - 0.21 = 1.79m
        wall_breach = (torch.abs(y_disks) >= 1.76).any(dim=3).any(dim=2).any(dim=1)
        cost_bounds = torch.sum(torch.relu(torch.abs(py) - 1.55) * 5000.0, dim=1)
        
        # Obstacles
        cost_obstacles = torch.zeros(self.K, device=self.device)
        cost_barrier = torch.zeros(self.K, device=self.device)
        obs_crash = torch.zeros(self.K, dtype=torch.bool, device=self.device)
        barrier_crash = torch.zeros(self.K, dtype=torch.bool, device=self.device)
        
        if obstacles_tensor is not None and len(obstacles_tensor) > 0:
            ox = obstacles_tensor[:, 0].view(1, 1, 1, -1)
            oy = obstacles_tensor[:, 1].view(1, 1, 1, -1)
            r_obs = obstacles_tensor[:, 2].view(1, 1, 1, -1)
            
            dist_sq = (x_disks - ox) ** 2 + (y_disks - oy) ** 2
            dists = torch.sqrt(torch.clamp(dist_sq, min=1e-4))
            margin = dists - (r_obs + r_disk)
            
            obs_penalties = torch.exp(-5.0 * torch.clamp(margin, min=0.0)) * torch.clamp(0.80 - margin, min=0.0)
            obs_crash = (margin <= 0.0).any(dim=3).any(dim=2).any(dim=1)
            cost_obstacles = torch.sum(obs_penalties, dim=(1, 2, 3)) * 25.0
            
            # Multi-Step Neural Barrier Network evaluation
            if barrier_net is not None:
                # Find obstacles ahead of vehicle within [ -0.5m, 6.0m ]
                px0 = states[0, 0, 0]
                dx_obs = obstacles_tensor[:, 0] - px0
                rel_mask = (dx_obs >= -0.5) & (dx_obs <= 6.0)
                rel_obs = obstacles_tensor[rel_mask]
                
                if len(rel_obs) > 0:
                    psi_flat = states[:, :, 2].reshape(-1, 1)
                    vx_flat = states[:, :, 3].reshape(-1, 1)
                    vy_flat = states[:, :, 4].reshape(-1, 1)
                    r_flat = states[:, :, 5].reshape(-1, 1)
                    
                    min_h_all = torch.full((self.K, self.horizon), 1.0, device=self.device)
                    
                    for obs_i in rel_obs:
                        r_eff = obs_i[2] + self.r_margin
                        scale = 0.60 / r_eff
                        dx_i = (states[:, :, 0] - obs_i[0]) * scale
                        dy_i = (states[:, :, 1] - obs_i[1]) * scale
                        
                        x_rel = torch.cat([dx_i.reshape(-1, 1), dy_i.reshape(-1, 1),
                                           psi_flat, vx_flat, vy_flat, r_flat], dim=1)
                        with torch.no_grad():
                            h_i = barrier_net(x_rel).view(self.K, self.horizon)
                        min_h_all = torch.minimum(min_h_all, h_i)
                    
                    # Barrier penalties
                    barrier_viol = torch.relu(-min_h_all)
                    barrier_near = torch.relu(0.20 - min_h_all)
                    cost_barrier = torch.sum(barrier_viol * 500.0 + barrier_near * 30.0, dim=1)
                    barrier_crash = (min_h_all < -0.30).any(dim=1)
        
        # Fatal crash: obstacle hit OR wall breach OR extreme barrier violation
        fatal_crash = obs_crash | wall_breach | barrier_crash
        cost_crash = fatal_crash.float() * 100000.0
        
        # 5. Control Effort with Heavy Braking Penalty (100x costlier than steering)
        ax = controls[:, :, 0]
        delta = controls[:, :, 1]
        braking = torch.relu(-ax)
        accel_pos = torch.relu(ax)
        
        cost_braking = torch.sum(braking ** 2, dim=1) * 5.0
        cost_accel = torch.sum(accel_pos ** 2, dim=1) * 0.05
        cost_steering = torch.sum(delta ** 2, dim=1) * 0.05
        cost_control = cost_braking + cost_accel + cost_steering
        
        total_costs = cost_progress + cost_speed + cost_centerline + cost_bounds + cost_heading + cost_obstacles + cost_barrier + cost_control + cost_crash
        return total_costs

def run_test(seed=175):
    env = ClutteredRacingEnv(num_obstacles=10, track_length=50.0, track_width=4.0, seed=seed)
    state = env.reset(start_vx=3.0)
    sc = NeuralICBFController(model_path=find_model(), mode="neural_icbf", r_margin=0.38, device="cuda")
    planner = MultiStepNICBFPlanner(num_samples=50, horizon=30, v_target=6.0, r_margin=0.38, device="cuda")
    
    current_u = np.array([env.actual_ax, env.actual_delta], dtype=np.float32)
    speeds = []
    outcome = "timeout"
    
    for step in range(1, 1000):
        obs_state = env.get_noisy_state()
        perceived = env.get_perceived_obstacles()
        
        u_nom = planner.plan(obs_state, perceived, barrier_net=sc.barrier_net)
        state, info = env.step(u_nom, dt=0.02)
        speeds.append(float(state[3]))
        
        if info["collision"]: outcome = "collision"; break
        if info["out_of_bounds"]: outcome = "out_of_bounds"; break
        if info["success"]: outcome = "success"; break
        
    print(f"Seed {seed} Multi-Step NICBF Alone -> outcome: {outcome}, steps: {step}, mean_speed: {np.mean(speeds):.2f}, final_x: {state[0]:.2f}")

run_test(175)

print("\n=== Running All 5 Seeds on Multi-Step NICBF ===")
for s in [101, 138, 175, 212, 249]:
    run_test(s)
