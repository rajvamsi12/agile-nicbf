import os
import sys
import numpy as np
import torch
import torch.nn as nn

class DynamicBicycleParams:
    """
    Default vehicle parameters from F1TENTH Gym / CommonRoad-style single-track dynamics.
    Source: F1TENTH Gym documentation, "Changing Parameters in Vehicle Dynamics".
    """
    def __init__(self,
                 mass=3.74,         # kg (total vehicle mass)
                 Iz=0.04712,        # kg*m^2 (yaw moment of inertia about z-axis)
                 lf=0.15875,        # m (CG to front axle)
                 lr=0.17145,        # m (CG to rear axle)
                 h=0.074,           # m (height of CG)
                 mu=1.0489,         # surface friction coefficient
                 C_Sf=4.718,        # cornering stiffness coefficient, front [1/rad]
                 C_Sr=5.4562,       # cornering stiffness coefficient, rear [1/rad]
                 g=9.81,            # m/s^2
                 v_min=0.2,         # m/s (min velocity threshold for slip angle division stability)
                 v_max=20.0,        # m/s (max longitudinal velocity)
                 a_max=9.51,        # m/s^2 (max acceleration/braking)
                 delta_max=0.4189,  # rad (~24 deg max steering angle)
                 sv_max=3.2,        # rad/s (max steering velocity)
                 width=0.31,        # m (vehicle width)
                 length=0.58):      # m (vehicle length)
        self.mass = mass
        self.Iz = Iz
        self.lf = lf
        self.lr = lr
        self.L = lf + lr
        self.h = h
        self.mu = mu
        self.C_Sf = C_Sf
        self.C_Sr = C_Sr
        self.g = g
        self.v_min = v_min
        self.v_max = v_max
        self.a_max = a_max
        self.delta_max = delta_max
        self.sv_max = sv_max
        self.width = width
        self.length = length
        
        # Static normal loads
        self.Fzf = mass * g * (lr / self.L)
        self.Fzr = mass * g * (lf / self.L)


class DynamicBicycleModel:
    """
    6D single-track dynamic bicycle model using F1TENTH parameter values.
    State x: [px, py, psi, vx, vy, r]
    Control u: [ax, delta]
    """
    def __init__(self, params=None, device='cpu'):
        self.p = params if params is not None else DynamicBicycleParams()
        self.device = device

    def continuous_dynamics_numpy(self, state, control):
        """
        Continuous time ODE: x_dot = f(x, u) in NumPy.
        State: [px, py, psi, vx, vy, r]
        Control: [ax, delta]
        """
        px, py, psi, vx, vy, r = state
        ax, delta = control
        
        vx_safe = np.maximum(vx, self.p.v_min)
        
        # Tire slip angles
        alpha_f = -np.arctan2(vy + self.p.lf * r, vx_safe) + delta
        alpha_r = -np.arctan2(vy - self.p.lr * r, vx_safe)
        
        # Dynamic Normal load transfer during acceleration/braking (F1TENTH vehicle_dynamics_st)
        Fzf_dyn = self.p.mass * self.p.g * (self.p.lr / self.p.L) - (self.p.mass * ax * self.p.h / self.p.L)
        Fzr_dyn = self.p.mass * self.p.g * (self.p.lf / self.p.L) + (self.p.mass * ax * self.p.h / self.p.L)
        Fzf_dyn = np.maximum(1.0, Fzf_dyn)
        Fzr_dyn = np.maximum(1.0, Fzr_dyn)
        
        # Lateral tire forces (Cornering stiffness * slip angle scaled by normal load)
        Fyf = self.p.mu * Fzf_dyn * self.p.C_Sf * alpha_f
        Fyr = self.p.mu * Fzr_dyn * self.p.C_Sr * alpha_r
        
        # Smooth Kamm's friction circle saturation: differentiable tanh preserves positive stiffness slope
        friction_rem = np.maximum(0.0, self.p.mu**2 - (ax / self.p.g)**2)
        max_Fyf = Fzf_dyn * np.sqrt(friction_rem)
        max_Fyr = Fzr_dyn * np.sqrt(friction_rem)
        Fyf = max_Fyf * np.tanh(Fyf / np.maximum(1e-3, max_Fyf))
        Fyr = max_Fyr * np.tanh(Fyr / np.maximum(1e-3, max_Fyr))
        
        # State derivatives
        d_px = vx * np.cos(psi) - vy * np.sin(psi)
        d_py = vx * np.sin(psi) + vy * np.cos(psi)
        d_psi = r
        d_vx = ax + vy * r - (Fyf * np.sin(delta)) / self.p.mass
        d_vy = -vx * r + (Fyf * np.cos(delta) + Fyr) / self.p.mass
        d_r = (self.p.lf * Fyf * np.cos(delta) - self.p.lr * Fyr) / self.p.Iz
        
        return np.array([d_px, d_py, d_psi, d_vx, d_vy, d_r], dtype=np.float32)

    def continuous_dynamics_torch(self, state, control):
        """
        Batched continuous time ODE for GPU MPPI and Neural Barrier Training.
        Matches the NumPy twin and applies smooth lateral tire-force saturation.
        state: [B, 6] -> (px, py, psi, vx, vy, r)
        control: [B, 2] -> (ax, delta)
        returns: [B, 6]
        """
        px = state[:, 0]
        py = state[:, 1]
        psi = state[:, 2]
        vx = state[:, 3]
        vy = state[:, 4]
        r = state[:, 5]
        
        ax = control[:, 0]
        delta = control[:, 1]
        
        vx_safe = torch.clamp(vx, min=self.p.v_min)
        
        alpha_f = -torch.atan2(vy + self.p.lf * r, vx_safe) + delta
        alpha_r = -torch.atan2(vy - self.p.lr * r, vx_safe)
        
        # Dynamic Normal load transfer
        Fzf_dyn = self.p.mass * self.p.g * (self.p.lr / self.p.L) - (self.p.mass * ax * self.p.h / self.p.L)
        Fzr_dyn = self.p.mass * self.p.g * (self.p.lf / self.p.L) + (self.p.mass * ax * self.p.h / self.p.L)
        Fzf_dyn = torch.clamp(Fzf_dyn, min=1.0)
        Fzr_dyn = torch.clamp(Fzr_dyn, min=1.0)
        
        # Lateral tire forces
        Fyf = self.p.mu * Fzf_dyn * self.p.C_Sf * alpha_f
        Fyr = self.p.mu * Fzr_dyn * self.p.C_Sr * alpha_r
        
        # Smooth Kamm's friction circle saturation: differentiable tanh prevents autograd zero-gradient cliffs
        friction_rem = torch.clamp(self.p.mu**2 - (ax / self.p.g)**2, min=0.0)
        max_Fyf = Fzf_dyn * torch.sqrt(friction_rem)
        max_Fyr = Fzr_dyn * torch.sqrt(friction_rem)
        Fyf = max_Fyf * torch.tanh(Fyf / torch.clamp(max_Fyf, min=1e-3))
        Fyr = max_Fyr * torch.tanh(Fyr / torch.clamp(max_Fyr, min=1e-3))
        
        d_px = vx * torch.cos(psi) - vy * torch.sin(psi)
        d_py = vx * torch.sin(psi) + vy * torch.cos(psi)
        d_psi = r
        d_vx = ax + vy * r - (Fyf * torch.sin(delta)) / self.p.mass
        d_vy = -vx * r + (Fyf * torch.cos(delta) + Fyr) / self.p.mass
        d_r = (self.p.lf * Fyf * torch.cos(delta) - self.p.lr * Fyr) / self.p.Iz
        
        return torch.stack([d_px, d_py, d_psi, d_vx, d_vy, d_r], dim=1)

    def step_rk4_torch(self, state, control, dt=0.02):
        """Vectorized RK4 integration on GPU with actuator limit guarding."""
        ax_safe = torch.clamp(control[:, 0], -self.p.a_max, self.p.a_max)
        delta_safe = torch.clamp(control[:, 1], -self.p.delta_max, self.p.delta_max)
        u_safe = torch.stack([ax_safe, delta_safe], dim=1)
        
        k1 = self.continuous_dynamics_torch(state, u_safe)
        k2 = self.continuous_dynamics_torch(state + 0.5 * dt * k1, u_safe)
        k3 = self.continuous_dynamics_torch(state + 0.5 * dt * k2, u_safe)
        k4 = self.continuous_dynamics_torch(state + dt * k3, u_safe)
        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        vx_clamped = torch.clamp(next_state[:, 3], min=0.0)
        is_zero = (vx_clamped <= 1e-4).unsqueeze(1)
        vy_clamped = torch.where(is_zero, torch.zeros_like(next_state[:, 4:5]), next_state[:, 4:5])
        r_clamped = torch.where(is_zero, torch.zeros_like(next_state[:, 5:6]), next_state[:, 5:6])
        return torch.cat([next_state[:, :3], vx_clamped.unsqueeze(1), vy_clamped, r_clamped], dim=1)

    def step_rk4_numpy(self, state, control, dt=0.02):
        """RK4 numerical integrator with actuator limit guarding."""
        u_safe = np.array([
            np.clip(control[0], -self.p.a_max, self.p.a_max),
            np.clip(control[1], -self.p.delta_max, self.p.delta_max)
        ], dtype=np.float32)
        
        # Physical stationary hold: vehicle at rest with brakes applied does not move
        if state[3] <= 1e-4 and u_safe[0] <= 0.0:
            next_state = state.copy()
            next_state[3] = 0.0
            next_state[4] = 0.0
            next_state[5] = 0.0
            return next_state
        
        k1 = self.continuous_dynamics_numpy(state, u_safe)
        k2 = self.continuous_dynamics_numpy(state + 0.5 * dt * k1, u_safe)
        k3 = self.continuous_dynamics_numpy(state + 0.5 * dt * k2, u_safe)
        k4 = self.continuous_dynamics_numpy(state + dt * k3, u_safe)
        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if next_state[3] <= 0.0:
            next_state[3] = 0.0
            next_state[4] = 0.0  # vy = 0 at standstill
            next_state[5] = 0.0  # r = 0 at standstill
            if state[3] <= 1e-4:
                next_state[0] = state[0]
                next_state[1] = state[1]
                next_state[2] = state[2]
        return next_state


class ClutteredRacingEnv:
    """
    Agile Multi-Obstacle Racing Environment matching F1TENTH Benchmark standards.
    Track width: 4.0 m
    Track length: 50.0 m
    Obstacles: N circular obstacles with radius r_obs.
    """
    def __init__(self, num_obstacles=20, track_length=50.0, track_width=4.0, seed=42):
        self.num_obstacles = num_obstacles
        self.track_length = track_length
        self.track_width = track_width
        self.model = DynamicBicycleModel()
        self.rng = np.random.RandomState(seed)
        self.obstacles = self._generate_obstacles()
        self.reset()

    def _generate_obstacles(self):
        """Generates random obstacle field with traversable gaps."""
        obstacles = []
        for _ in range(self.num_obstacles):
            ox = self.rng.uniform(6.0, self.track_length - 5.0)
            oy = self.rng.uniform(-self.track_width / 2.0 + 0.6, self.track_width / 2.0 - 0.6)
            r_obs = self.rng.uniform(0.3, 0.50) # 30cm - 50cm radius
            obstacles.append([ox, oy, r_obs])
        return np.array(obstacles, dtype=np.float32)

    def reset(self, start_vx=2.0):
        # state: [px, py, psi, vx, vy, r]
        self.state = np.array([0.0, 0.0, 0.0, start_vx, 0.0, 0.0], dtype=np.float32)
        self.t = 0.0
        return self.state.copy()

    def step(self, control, dt=0.02):
        self.state = self.model.step_rk4_numpy(self.state, control, dt=dt)
        self.t += dt
        
        px, py = self.state[0], self.state[1]
        collision = False
        min_dist = float('inf')
        
        for ox, oy, r_obs in self.obstacles:
            dist = np.sqrt((px - ox)**2 + (py - oy)**2) - r_obs - 0.2 # 20cm car margin
            if dist < min_dist:
                min_dist = dist
            if dist <= 0.0:
                collision = True
                break
                
        out_of_bounds = abs(py) > (self.track_width / 2.0)
        done = (self.state[0] >= self.track_length) or collision or out_of_bounds
        
        info = {
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "min_obstacle_distance": min_dist,
            "success": (self.state[0] >= self.track_length) and not collision and not out_of_bounds
        }
        return self.state.copy(), info
