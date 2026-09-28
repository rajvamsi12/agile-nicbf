import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Universal import
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dynamic_bicycle_env import ClutteredRacingEnv
from mppi_planner import MPPIPlanner
from safe_controller import NeuralICBFController

def run_and_record_trajectory(method, seed=138, max_steps=500, dt=0.02, device="cpu"):
    """
    Runs an episode and records detailed telemetry: x, y, psi, vx, vy, r, ax, delta, min_d.
    """
    env = ClutteredRacingEnv(num_obstacles=20, track_length=50.0, track_width=4.0, seed=seed)
    state = env.reset(start_vx=3.0)
    
    planner = MPPIPlanner(num_samples=1000, horizon=35, v_target=6.0, device=device)
    shield = None
    if method in ['analytical_hocbf', 'neural_icbf']:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "neural_icbf_6d.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "neural_icbf_6d.pth")
        shield = NeuralICBFController(model_path=model_path, mode=method, r_margin=0.26, alpha1=2.5, alpha2=3.5, alpha_w=3.5, device=device)

    current_u = np.array([2.0, 0.0], dtype=np.float32)
    history = {
        "x": [state[0]], "y": [state[1]], "psi": [state[2]],
        "vx": [state[3]], "vy": [state[4]], "r": [state[5]],
        "ax": [current_u[0]], "delta": [current_u[1]],
        "min_dist": [], "qp_active": []
    }
    
    done = False
    step = 0
    while not done and step < max_steps:
        px, py = state[0], state[1]
        nearby_obs = []
        for ox, oy, ro in env.obstacles:
            if -2.0 <= (ox - px) <= 15.0 and abs(oy - py) <= 3.0:
                nearby_obs.append([ox, oy, ro])
        nearby_obs = np.array(nearby_obs, dtype=np.float32) if len(nearby_obs) > 0 else np.zeros((0, 3), dtype=np.float32)
        
        u_nom = planner.plan(state, nearby_obs)
        if shield is not None:
            u_safe, qp_info = shield.filter_control(state, current_u, u_nom, nearby_obs, dt=dt)
            u_exec = u_safe
            history["qp_active"].append(qp_info["qp_active"])
        else:
            u_exec = u_nom
            history["qp_active"].append(False)
            
        current_u = u_exec
        state, info = env.step(u_exec, dt=dt)
        step += 1
        
        history["x"].append(state[0])
        history["y"].append(state[1])
        history["psi"].append(state[2])
        history["vx"].append(state[3])
        history["vy"].append(state[4])
        history["r"].append(state[5])
        history["ax"].append(u_exec[0])
        history["delta"].append(u_exec[1])
        history["min_dist"].append(info["min_obstacle_distance"])
        
        if info["collision"] or info["out_of_bounds"] or info["success"]:
            done = True
            
    return history, env.obstacles, info

def plot_trajectory_comparison(seed=138, save_path="results/trajectory_comparison.png"):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    methods = [
        ("mppi_alone", "MPPI Alone", "gray", ":"),
        ("analytical_hocbf", "Analytical HOCBF-QP", "#d95f02", "--"),
        ("neural_icbf", "Agile-NICBF (Ours)", "#1b9e77", "-")
    ]
    
    obstacles = None
    all_hist = {}
    
    for method, label, color, ls in methods:
        hist, obs, info = run_and_record_trajectory(method, seed=seed)
        all_hist[method] = hist
        if obstacles is None:
            obstacles = obs
            
    # 1. 2D Trajectory Map
    ax_map = axes[0]
    # Track Boundaries
    ax_map.axhline(2.0, color='black', linewidth=2.0, label="Track Wall ($W = \\pm 2.0\\text{m}$)")
    ax_map.axhline(-2.0, color='black', linewidth=2.0)
    
    # Obstacles
    for i, (ox, oy, ro) in enumerate(obstacles):
        # Physical obstacle
        circle = patches.Circle((ox, oy), ro, edgecolor='#b30000', facecolor='#fbb4ae', alpha=0.8, linewidth=1.5)
        ax_map.add_patch(circle)
        # 0.20m crash boundary
        crash_circle = patches.Circle((ox, oy), ro + 0.20, edgecolor='#e41a1c', facecolor='none', linestyle=':', linewidth=1.0)
        ax_map.add_patch(crash_circle)
        
    for method, label, color, ls in methods:
        hist = all_hist[method]
        ax_map.plot(hist["x"], hist["y"], label=label, color=color, linestyle=ls, linewidth=2.2)
        # Mark endpoints
        marker = 'o' if hist["x"][-1] >= 50.0 else 'x'
        ax_map.scatter(hist["x"][-1], hist["y"][-1], color=color, s=80, marker=marker, zorder=5)

    ax_map.set_xlim(-1.0, 52.0)
    ax_map.set_ylim(-2.5, 2.5)
    ax_map.set_xlabel("Track Longitudinal Position $x$ [m]", fontsize=11, fontweight='bold')
    ax_map.set_ylabel("Lateral Position $y$ [m]", fontsize=11, fontweight='bold')
    ax_map.set_title(f"High-Speed Autonomous Racing Trajectories (Seed {seed}, Target Speed 6.0 m/s)", fontsize=13, fontweight='bold')
    ax_map.legend(loc='upper right', frameon=True, fontsize=10)
    
    # 2. Longitudinal Speed Profile vx(t)
    ax_vx = axes[1]
    for method, label, color, ls in methods:
        hist = all_hist[method]
        t = np.arange(len(hist["vx"])) * 0.02
        ax_vx.plot(t, hist["vx"], label=label, color=color, linestyle=ls, linewidth=1.8)
    ax_vx.axhline(6.0, color='black', linestyle=':', label="Target $v_{\\text{target}}=6\\text{m/s}$")
    ax_vx.set_ylabel("Speed $v_x$ [m/s]", fontsize=11, fontweight='bold')
    ax_vx.set_title("Longitudinal Velocity Tracking & Braking Profiles", fontsize=11, fontweight='bold')
    ax_vx.set_ylim(0.0, 7.5)
    ax_vx.legend(loc='lower right', frameon=True, fontsize=9)

    # 3. Minimum Obstacle Clearance Profile
    ax_dist = axes[2]
    for method, label, color, ls in methods:
        hist = all_hist[method]
        if len(hist["min_dist"]) > 0:
            t = np.arange(len(hist["min_dist"])) * 0.02
            ax_dist.plot(t, hist["min_dist"], label=label, color=color, linestyle=ls, linewidth=1.8)
    ax_dist.axhline(0.0, color='red', linestyle='--', linewidth=1.5, label="Collision Threshold (0.0 m)")
    ax_dist.set_xlabel("Time $t$ [seconds]", fontsize=11, fontweight='bold')
    ax_dist.set_ylabel("Clearance $d_{\\min}$ [m]", fontsize=11, fontweight='bold')
    ax_dist.set_title("Minimum Clearance to Obstacle Boundary", fontsize=11, fontweight='bold')
    ax_dist.set_ylim(-0.1, 2.5)
    ax_dist.legend(loc='upper right', frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Stage 6] Saved Trajectory Comparison Plot to {save_path}")

if __name__ == "__main__":
    plot_trajectory_comparison(seed=138)
