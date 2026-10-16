import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# Universal path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dynamic_bicycle_env import ClutteredRacingEnv, DynamicBicycleParams
from mppi_planner import MPPIPlanner
from safe_controller import NeuralICBFController


def record_simulation(method='neural_icbf', seed=138, max_steps=500, dt=0.02, device="cpu"):
    """
    Runs simulation and records step-by-step telemetry for MP4 video rendering.
    """
    env = ClutteredRacingEnv(num_obstacles=20, track_length=50.0, track_width=4.0, seed=seed)
    state = env.reset(start_vx=3.0)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "neural_icbf_6d.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "neural_icbf_6d.pth")

    planner = MPPIPlanner(num_samples=1000, horizon=35, v_target=6.0, device=device)
    shield = None
    if method in ['analytical_hocbf', 'neural_icbf']:
        shield = NeuralICBFController(model_path=model_path, mode=method, r_margin=0.26, alpha1=2.5, alpha2=3.5, alpha_w=3.5, device=device)

    current_u = np.array([2.0, 0.0], dtype=np.float32)
    frames = []
    
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
            qp_active = qp_info["qp_active"]
            h_val = qp_info["min_h_val"]
        else:
            u_exec = u_nom
            qp_active = False
            h_val = 1.0

        frames.append({
            "step": step,
            "time": step * dt,
            "x": state[0],
            "y": state[1],
            "psi": state[2],
            "vx": state[3],
            "vy": state[4],
            "r": state[5],
            "ax_nom": u_nom[0],
            "delta_nom": u_nom[1],
            "ax_exec": u_exec[0],
            "delta_exec": u_exec[1],
            "qp_active": qp_active,
            "h_val": h_val,
            "min_dist": 0.0
        })

        current_u = u_exec
        state, info = env.step(u_exec, dt=dt)
        frames[-1]["min_dist"] = info["min_obstacle_distance"]
        step += 1

        if info["collision"] or info["out_of_bounds"] or info["success"]:
            done = True

    return frames, env.obstacles


def create_mp4_video(frames, obstacles, output_mp4="results/agile_nicbf_demo.mp4", fps=30):
    """
    Renders high-quality MP4 animation of the car navigating the cluttered track.
    """
    print(f"[Video Generator] Rendering {len(frames)} frames to {output_mp4} at {fps} FPS...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9), dpi=150)
    
    # 2-Row Layout: Top = Track View, Bottom = Live Telemetry Graphs
    gs = fig.add_gridspec(3, 3, height_ratios=[1.8, 1.0, 1.0], hspace=0.35, wspace=0.25)
    
    ax_track = fig.add_subplot(gs[0, :])
    ax_speed = fig.add_subplot(gs[1, 0])
    ax_steer = fig.add_subplot(gs[1, 1])
    ax_dist = fig.add_subplot(gs[1, 2])
    ax_accel = fig.add_subplot(gs[2, 0])
    ax_hval = fig.add_subplot(gs[2, 1])
    ax_status = fig.add_subplot(gs[2, 2])
    
    # Extract timeseries
    times = [f["time"] for f in frames]
    vxs = [f["vx"] for f in frames]
    steers_nom = [f["delta_nom"] * 180 / np.pi for f in frames]
    steers_exec = [f["delta_exec"] * 180 / np.pi for f in frames]
    accels_nom = [f["ax_nom"] for f in frames]
    accels_exec = [f["ax_exec"] for f in frames]
    dists = [f["min_dist"] for f in frames]
    h_vals = [f["h_val"] for f in frames]
    qp_acts = [f["qp_active"] for f in frames]

    # Car dimensions (F1TENTH)
    car_len = 0.58
    car_wid = 0.31

    def update(frame_idx):
        f = frames[frame_idx]
        t_curr = f["time"]

        # ---------------- 1. Top Track View ----------------
        ax_track.clear()
        ax_track.set_facecolor('#121212')
        
        # Track Walls (W = +-2.0m)
        ax_track.axhline(2.0, color='#e74c3c', linewidth=3, label='Track Wall (W = $\\pm 2.0$m)')
        ax_track.axhline(-2.0, color='#e74c3c', linewidth=3)
        ax_track.axhline(0.0, color='#444444', linestyle='--', linewidth=1.0) # Centerline

        # Finish line
        ax_track.axvline(50.0, color='#2ecc71', linewidth=3, linestyle='--', label='Finish Line (50m)')

        # Obstacles
        for ox, oy, ro in obstacles:
            # Physical Obstacle
            obs_patch = patches.Circle((ox, oy), ro, edgecolor='#e74c3c', facecolor='#c0392b', alpha=0.9, zorder=3)
            ax_track.add_patch(obs_patch)
            # 0.20m crash boundary
            buf_patch = patches.Circle((ox, oy), ro + 0.20, edgecolor='#f39c12', facecolor='none', linestyle=':', linewidth=1.2, zorder=2)
            ax_track.add_patch(buf_patch)

        # Vehicle Body (Oriented Box)
        px, py, psi = f["x"], f["y"], f["psi"]
        cos_p, sin_p = np.cos(psi), np.sin(psi)
        
        # 4 Corners of car
        dx_f = car_len / 2.0
        dx_r = -car_len / 2.0
        dy_l = car_wid / 2.0
        dy_r = -car_wid / 2.0
        
        corners_x = [px + dx_f*cos_p - dy_l*sin_p, px + dx_f*cos_p - dy_r*sin_p,
                     px + dx_r*cos_p - dy_r*sin_p, px + dx_r*cos_p - dy_l*sin_p]
        corners_y = [py + dx_f*sin_p + dy_l*cos_p, py + dx_f*sin_p + dy_r*cos_p,
                     py + dx_r*sin_p + dy_r*cos_p, py + dx_r*sin_p + dy_l*cos_p]
        
        car_body = patches.Polygon(np.column_stack([corners_x, corners_y]), closed=True, 
                                   edgecolor='#00ffcc', facecolor='#1abc9c', alpha=0.9, zorder=5)
        ax_track.add_patch(car_body)

        # Front Wheels with Steering Angle
        delta = f["delta_exec"]
        wheel_len = 0.18
        # Front Left Wheel
        fl_x, fl_y = corners_x[0], corners_y[0]
        fl_ang = psi + delta
        ax_track.plot([fl_x - (wheel_len/2)*np.cos(fl_ang), fl_x + (wheel_len/2)*np.cos(fl_ang)],
                      [fl_y - (wheel_len/2)*np.sin(fl_ang), fl_y + (wheel_len/2)*np.sin(fl_ang)],
                      color='#f1c40f', linewidth=3, zorder=6)
        # Front Right Wheel
        fr_x, fr_y = corners_x[1], corners_y[1]
        ax_track.plot([fr_x - (wheel_len/2)*np.cos(fl_ang), fr_x + (wheel_len/2)*np.cos(fl_ang)],
                      [fr_y - (wheel_len/2)*np.sin(fl_ang), fr_y + (wheel_len/2)*np.sin(fl_ang)],
                      color='#f1c40f', linewidth=3, zorder=6)

        # Past Trajectory Breadcrumb
        past_x = [fr["x"] for fr in frames[:frame_idx+1]]
        past_y = [fr["y"] for fr in frames[:frame_idx+1]]
        ax_track.plot(past_x, past_y, color='#00ffcc', linewidth=1.8, linestyle='-', alpha=0.7, zorder=4)

        # Dynamic Scrolling Camera centered around car (window = 14m)
        x_min = max(-1.0, px - 3.0)
        x_max = x_min + 16.0
        ax_track.set_xlim(x_min, x_max)
        ax_track.set_ylim(-2.6, 2.6)
        ax_track.set_aspect('equal')
        ax_track.set_xlabel("Track Distance X [m]", fontsize=11, fontweight='bold')
        ax_track.set_ylabel("Track Width Y [m]", fontsize=11, fontweight='bold')
        ax_track.set_title(f"Agile-NICBF High-Speed Autonomous Racing (Time: {t_curr:.2f}s | Speed: {f['vx']:.2f} m/s | Pos: x={px:.1f}m, y={py:.2f}m)", fontsize=13, fontweight='bold', color='#00ffcc')

        # ---------------- 2. Live Telemetry Panels ----------------
        # A. Speed Profile
        ax_speed.clear()
        ax_speed.plot(times[:frame_idx+1], vxs[:frame_idx+1], color='#00ffcc', linewidth=2.0, label='Actual $v_x$')
        ax_speed.axhline(6.0, color='#e74c3c', linestyle=':', label='Target 6.0 m/s')
        ax_speed.set_xlim(0, max(times[-1], 1.0))
        ax_speed.set_ylim(0, 7.5)
        ax_speed.set_title(f"Longitudinal Speed: {f['vx']:.2f} m/s", fontsize=10, fontweight='bold')
        ax_speed.set_ylabel("Speed [m/s]", fontsize=9)
        ax_speed.legend(loc='lower right', fontsize=8)

        # B. Steering Angle (Nominal vs Filtered)
        ax_steer.clear()
        ax_steer.plot(times[:frame_idx+1], steers_nom[:frame_idx+1], color='gray', linestyle=':', label='MPPI Nom $\\delta$')
        ax_steer.plot(times[:frame_idx+1], steers_exec[:frame_idx+1], color='#f39c12', linewidth=2.0, label='Safe $\\delta$')
        ax_steer.set_xlim(0, max(times[-1], 1.0))
        ax_steer.set_ylim(-26, 26)
        ax_steer.set_title(f"Steering Angle: {f['delta_exec']*180/np.pi:.1f}°", fontsize=10, fontweight='bold')
        ax_steer.set_ylabel("Steer [deg]", fontsize=9)
        ax_steer.legend(loc='lower right', fontsize=8)

        # C. Minimum Obstacle Clearance
        ax_dist.clear()
        ax_dist.plot(times[:frame_idx+1], dists[:frame_idx+1], color='#3498db', linewidth=2.0)
        ax_dist.axhline(0.0, color='#e74c3c', linestyle='--', label='Crash Threshold')
        ax_dist.set_xlim(0, max(times[-1], 1.0))
        ax_dist.set_ylim(-0.1, 2.5)
        ax_dist.set_title(f"Min Obstacle Clearance: {f['min_dist']:.2f} m", fontsize=10, fontweight='bold')
        ax_dist.set_ylabel("Margin [m]", fontsize=9)
        ax_dist.legend(loc='upper right', fontsize=8)

        # D. Acceleration
        ax_accel.clear()
        ax_accel.plot(times[:frame_idx+1], accels_nom[:frame_idx+1], color='gray', linestyle=':', label='Nom $a_x$')
        ax_accel.plot(times[:frame_idx+1], accels_exec[:frame_idx+1], color='#2ecc71', linewidth=2.0, label='Safe $a_x$')
        ax_accel.set_xlim(0, max(times[-1], 1.0))
        ax_accel.set_ylim(-11, 11)
        ax_accel.set_title(f"Acceleration $a_x$: {f['ax_exec']:.2f} m/s²", fontsize=10, fontweight='bold')
        ax_accel.set_xlabel("Time [s]", fontsize=9)
        ax_accel.set_ylabel("Accel [m/s²]", fontsize=9)
        ax_accel.legend(loc='lower right', fontsize=8)

        # E. Barrier Margin
        ax_hval.clear()
        ax_hval.plot(times[:frame_idx+1], h_vals[:frame_idx+1], color='#9b59b6', linewidth=2.0)
        ax_hval.axhline(0.0, color='#e74c3c', linestyle='--', label='Safety Invariance $h=0$')
        ax_hval.set_xlim(0, max(times[-1], 1.0))
        ax_hval.set_ylim(-1.2, 1.2)
        ax_hval.set_title(f"Barrier Metric $h_\\theta(x)$: {f['h_val']:.2f}", fontsize=10, fontweight='bold')
        ax_hval.set_xlabel("Time [s]", fontsize=9)
        ax_hval.set_ylabel("Safety Metric", fontsize=9)
        ax_hval.legend(loc='lower right', fontsize=8)

        # F. Live Safety Shield Status Card
        ax_status.clear()
        ax_status.axis('off')
        if f["qp_active"]:
            box_color = '#e67e22'
            status_text = "SAFETY SHIELD ACTIVE\n(QP Intervening)"
        else:
            box_color = '#27ae60'
            status_text = "NOMINAL RACING\n(Zero Intervention)"
            
        status_box = patches.FancyBboxPatch((0.05, 0.15), 0.9, 0.7, boxstyle="round,pad=0.1",
                                            facecolor=box_color, edgecolor='white', linewidth=2.0, alpha=0.85)
        ax_status.add_patch(status_box)
        ax_status.text(0.5, 0.5, status_text, fontsize=11, fontweight='bold',
                       ha='center', va='center', color='white')

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=int(1000/fps), blit=False)
    
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Agile-NICBF'), bitrate=2500)
    ani.save(output_mp4, writer=writer)
    plt.close()
    print(f"[Video Generator] Successfully rendered MP4 video to {output_mp4}")


if __name__ == "__main__":
    frames, obstacles = record_simulation(method='neural_icbf', seed=138)
    create_mp4_video(frames, obstacles, output_mp4="results/agile_nicbf_demo.mp4")
