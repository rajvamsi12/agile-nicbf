import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dynamic_bicycle_env import ClutteredRacingEnv, DynamicBicycleParams
from mppi_planner import MPPIPlanner
from safe_controller import NeuralICBFController

import torch

def plot_kamm_friction_circle(seed=138, save_path="results/friction_circle_compliance.png"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    params = DynamicBicycleParams()
    mu = params.mu
    g = params.g
    a_max = params.a_max
    
    env = ClutteredRacingEnv(num_obstacles=10, track_length=50.0, track_width=4.0, seed=seed)
    state = env.reset(start_vx=3.0)
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "neural_icbf_6d.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "neural_icbf_6d.pth")
        
    planner = MPPIPlanner(num_samples=100, horizon=30, v_target=6.0, r_margin=0.26, device="cpu")
    shield = NeuralICBFController(model_path=model_path, mode='neural_icbf', r_margin=0.26, alpha1=2.5, alpha2=3.5, alpha_w=2.5, device="cpu")
    
    current_u = np.array([env.actual_ax, env.actual_delta], dtype=np.float32)
    ax_nom_list, ay_nom_list = [], []
    ax_safe_list, ay_safe_list = [], []
    
    done = False
    step = 0
    while not done and step < 750:
        perceived_obs = env.get_perceived_obstacles()
        obs_state = env.get_noisy_state()
        
        u_nom = planner.plan(obs_state, perceived_obs)
        u_safe, qp_info = shield.filter_control(obs_state, current_u, u_nom, perceived_obs, dt=0.02)
        
        # Nominal vehicle acceleration
        f_nom = env.model.continuous_dynamics_numpy(state, u_nom)
        ax_nom = u_nom[0]
        ay_nom = f_nom[4] + state[3] * state[5] # ay = vy_dot + vx * r
        ax_nom_list.append(ax_nom)
        ay_nom_list.append(ay_nom)
        
        # Shielded vehicle acceleration
        f_safe = env.model.continuous_dynamics_numpy(state, u_safe)
        ax_safe = u_safe[0]
        ay_safe = f_safe[4] + state[3] * state[5]
        ax_safe_list.append(ax_safe)
        ay_safe_list.append(ay_safe)
        
        state, info = env.step(u_safe, dt=0.02)
        current_u = info["actual_u"]
        step += 1
        
        if info["collision"] or info["out_of_bounds"] or info["success"]:
            done = True

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    mu_g = mu * g
    theta = np.linspace(0, 2*np.pi, 200)
    circle_x = mu_g * np.cos(theta)
    circle_y = mu_g * np.sin(theta)
    
    # 1. Unshielded / Nominal Plan Acceleration
    ax1 = axes[0]
    ax1.plot(circle_x, circle_y, 'r--', linewidth=2.0, label=f"Tire Saturation Limit ($\\mu g = {mu_g:.2f}\\,\\mathrm{{m/s^2}}$)")
    ax1.scatter(ax_nom_list, ay_nom_list, color='gray', alpha=0.6, s=25, label="Nominal MPPI Commands")
    ax1.set_xlabel("Longitudinal Acceleration $a_x$ [$\\mathrm{m/s^2}$]", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Lateral Acceleration $a_y$ [$\\mathrm{m/s^2}$]", fontsize=11, fontweight='bold')
    ax1.set_title("Nominal MPPI Dynamic Load", fontsize=12, fontweight='bold')
    ax1.set_xlim(-12.0, 12.0)
    ax1.set_ylim(-12.0, 12.0)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    
    # 2. Agile-NICBF Shielded Acceleration
    ax2 = axes[1]
    ax2.plot(circle_x, circle_y, 'r--', linewidth=2.0, label=f"Tire Saturation Limit ($\\mu g = {mu_g:.2f}\\,\\mathrm{{m/s^2}}$)")
    ax2.scatter(ax_safe_list, ay_safe_list, color='#1b9e77', alpha=0.7, s=25, label="Agile-NICBF Actuation ($u_{\\text{safe}}$)")
    ax2.set_xlabel("Longitudinal Acceleration $a_x$ [$\\mathrm{m/s^2}$]", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Lateral Acceleration $a_y$ [$\\mathrm{m/s^2}$]", fontsize=11, fontweight='bold')
    ax2.set_title("Agile-NICBF Acceleration Distribution", fontsize=12, fontweight='bold')
    ax2.set_xlim(-12.0, 12.0)
    ax2.set_ylim(-12.0, 12.0)
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Stage 6] Saved Acceleration Distribution Plot to {save_path}")

if __name__ == "__main__":
    plot_kamm_friction_circle(seed=138)
