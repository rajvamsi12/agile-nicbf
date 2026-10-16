import os
import sys
import time
import argparse
import json
import numpy as np
import torch

# Universal path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dynamic_bicycle_env import ClutteredRacingEnv
from mppi_planner import MPPIPlanner
from safe_controller import NeuralICBFController


def get_git_commit():
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"

def wilson_score_interval(successes, n, confidence=0.95):
    """Computes Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.959964 if confidence == 0.95 else 2.575829
    p = float(successes) / float(n)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * (n**2)))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return float(low * 100.0), float(high * 100.0)

def run_single_episode(method, seed, max_steps=750, dt=0.02, v_target=6.0, device="cpu"):
    """
    Runs a single closed-loop navigation episode on ClutteredRacingEnv.
    Standard Benchmark Suite:
    1. 'mppi_alone'          (Williams et al., IEEE T-RO 2018) - Nominal Real-time Planner (K=100)
    2. 'analytical_hocbf'    (Xiao & Belta, IEEE TAC 2022)    - 2nd-Order HOCBF Safety Filter
    3. 'neural_icbf'         (Decoupled Neural ICBF)          - Nominal MPPI + Neural ICBF Safety Shield
    4. 'agile_nicbf_coupled' (Agile-NICBF - Ours)             - Barrier-guided MPPI + Neural ICBF Shield
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = ClutteredRacingEnv(num_obstacles=10, track_length=50.0, track_width=4.0, seed=seed)
    state = env.reset(start_vx=3.0)
    
    # Initialize appropriate controller / planner
    mppi_planner = None
    shield = None
    mpcc_controller = None
    sfc_planner = None
    drl_policy = None
    
    if method in ['mppi_alone', 'analytical_hocbf', 'neural_icbf', 'agile_nicbf_coupled']:
        # Standard real-time sample budget (K=100 samples)
        mppi_planner = MPPIPlanner(num_samples=500 if str(device) == "cuda" else 100, horizon=30, v_target=v_target, r_margin=0.38, device=device)
        
    if method in ['analytical_hocbf', 'neural_icbf', 'agile_nicbf_coupled']:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "neural_icbf_6d.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "neural_icbf_6d.pth")
        shield_mode = 'neural_icbf' if method in ['neural_icbf', 'agile_nicbf_coupled'] else 'analytical_hocbf'
        shield = NeuralICBFController(model_path=model_path, mode=shield_mode, r_margin=0.38, alpha1=2.5, alpha2=3.5, alpha_w=2.5, device=device)
        
    elif method == 'mpcc_liniger':
        from mpcc_controller import MPCCController
        mpcc_controller = MPCCController(horizon=20, dt=dt, v_target=v_target, device=device)
        
    elif method == 'sfc_minco':
        from sfc_planner import SFCPolynomialPlanner
        sfc_planner = SFCPolynomialPlanner(horizon_dist=12.0, num_corridors=5, v_target=v_target)
        
    elif method == 'drl_policy':
        from drl_racing_policy import DRLRacingPolicy
        drl_policy = DRLRacingPolicy(v_target=v_target, device=device)

    current_u = np.array([env.actual_ax, env.actual_delta], dtype=np.float32)
    step_count = 0
    speeds = []
    planner_latencies = []
    qp_latencies = []
    total_latencies = []
    interventions = 0
    done = False
    outcome = "timeout"
    
    start_time = time.time()
    
    while not done and step_count < max_steps:
        # Realistic Perception & State Estimation:
        # 1. LiDAR 7.0m sensing horizon & 270 deg FOV (Evans et al. ICRA 2023)
        perceived_obs = env.get_perceived_obstacles()
        # 2. EKF State estimation with sensor noise (Betz et al. IEEE T-IV 2023)
        obs_state = env.get_noisy_state()
        
        # 1. Compute Control Action per Method
        t0 = time.time()
        mppi_lat = 0.0
        qp_lat = 0.0
        
        if method == 'mppi_alone':
            t_plan = time.time()
            u_exec = mppi_planner.plan(obs_state, perceived_obs)
            mppi_lat = (time.time() - t_plan) * 1000.0
            
        elif method in ['analytical_hocbf', 'neural_icbf']:
            t_plan = time.time()
            # Fair, decoupled evaluation: both shields filter identical nominal MPPI plans
            u_nom = mppi_planner.plan(obs_state, perceived_obs)
            mppi_lat = (time.time() - t_plan) * 1000.0
            
            t_qp = time.time()
            u_safe, qp_info = shield.filter_control(obs_state, current_u, u_nom, perceived_obs, dt=dt)
            qp_lat = (time.time() - t_qp) * 1000.0
            if qp_info["qp_active"]:
                interventions += 1
            u_exec = u_safe
            
        elif method == 'agile_nicbf_coupled':
            t_plan = time.time()
            # Coupled evaluation: MPPI uses barrier net cost guidance
            u_nom = mppi_planner.plan(obs_state, perceived_obs, barrier_net=shield.barrier_net)
            mppi_lat = (time.time() - t_plan) * 1000.0
            
            t_qp = time.time()
            u_safe, qp_info = shield.filter_control(obs_state, current_u, u_nom, perceived_obs, dt=dt)
            qp_lat = (time.time() - t_qp) * 1000.0
            if qp_info["qp_active"]:
                interventions += 1
            u_exec = u_safe
            
        elif method == 'mpcc_liniger':
            u_exec, info_m = mpcc_controller.solve(obs_state, perceived_obs)
            mppi_lat = info_m["latency_ms"]
            
        elif method == 'sfc_minco':
            u_exec, info_s = sfc_planner.plan(obs_state, perceived_obs)
            mppi_lat = info_s["latency_ms"]
            
        elif method == 'drl_policy':
            u_exec, info_d = drl_policy.plan(obs_state, perceived_obs)
            mppi_lat = info_d["latency_ms"]
            
        total_lat = (time.time() - t0) * 1000.0
        planner_latencies.append(mppi_lat)
        qp_latencies.append(qp_lat)
        total_latencies.append(total_lat)
        
        state, info = env.step(u_exec, dt=dt)
        current_u = info["actual_u"]
        step_count += 1
        speeds.append(float(state[3]))
        
        if info["collision"]:
            outcome = "collision"
            done = True
        elif info["out_of_bounds"]:
            outcome = "out_of_bounds"
            done = True
        elif info["success"]:
            outcome = "success"
            done = True
            
    traj_time = step_count * dt
    
    return {
        "method": method,
        "seed": seed,
        "outcome": outcome,
        "success": outcome == "success",
        "collision": outcome == "collision",
        "out_of_bounds": outcome == "out_of_bounds",
        "steps": step_count,
        "traj_time_s": traj_time,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "max_speed": float(np.max(speeds)) if speeds else 0.0,
        "interventions": interventions,
        "intervention_rate": float(interventions / max(1, step_count)),
        "mean_planner_latency_ms": float(np.mean(planner_latencies)) if planner_latencies else 0.0,
        "mean_qp_latency_ms": float(np.mean([lat for lat in qp_latencies if lat > 0])) if any(lat > 0 for lat in qp_latencies) else 0.0,
        "mean_total_latency_ms": float(np.mean(total_latencies)) if total_latencies else 0.0,
        "mean_latency_ms": float(np.mean(total_latencies)) if total_latencies else 0.0,
        "final_x": float(state[0]),
        "final_y": float(state[1]),
    }


def run_benchmark(num_episodes=50, max_steps=750, methods=None, seeds=None, device="cpu", save_dir="./results"):
    """
    Executes comparative evaluation across the 4 benchmark methods with Wilson confidence intervals.
    """
    if methods is None:
        methods = ["mppi_alone", "analytical_hocbf", "neural_icbf", "agile_nicbf_coupled"]
    if seeds is None:
        seeds = [101 + i * 37 for i in range(num_episodes)]
        
    print(f"\n{'='*95}")
    print(f"  4-Method Autonomous Racing Benchmark Suite")
    print(f"  Episodes: {len(seeds)} per method | Max Steps: {max_steps} ({max_steps*0.02:.1f}s) | Device: {device}")
    print(f"  Methods: {', '.join(methods)}")
    print(f"{'='*95}\n")
    
    all_results = {m: [] for m in methods}
    
    for method in methods:
        print(f"\n--- Evaluating Baseline: {method} ({len(seeds)} episodes) ---")
        for i, seed in enumerate(seeds):
            res = run_single_episode(method, seed, max_steps=max_steps, device=device)
            all_results[method].append(res)
            print(f"  [{i+1:03d}/{len(seeds):03d}] Seed {seed:4d} -> Outcome: {res['outcome']:13s} | Steps: {res['steps']:3d} ({res['traj_time_s']:.2f}s) | Speed: {res['mean_speed']:.2f} m/s | QP: {res['mean_qp_latency_ms']:.2f}ms | Total: {res['mean_total_latency_ms']:.2f}ms")
            
    # Compute Aggregated Metrics
    summary = {}
    print(f"\n\n{'='*120}")
    print(f"{'Method':<22} | {'Success (95% CI)':<20} | {'Collision (95% CI)':<20} | {'OOB':<6} | {'Speed (m/s)':<11} | {'Time (s)':<8} | {'QP Lat':<9} | {'Total Lat':<10}")
    print(f"{'-'*120}")
    
    for method in methods:
        res_list = all_results[method]
        n = len(res_list)
        successes = sum(1 for r in res_list if r["success"])
        collisions = sum(1 for r in res_list if r["collision"])
        oob = sum(1 for r in res_list if r["out_of_bounds"])
        
        succ_res = [r for r in res_list if r["success"]]
        avg_speed = np.mean([r["mean_speed"] for r in res_list]) if res_list else 0.0
        avg_time = np.mean([r["traj_time_s"] for r in succ_res]) if succ_res else 0.0
        avg_planner_lat = np.mean([r.get("mean_planner_latency_ms", 0.0) for r in res_list]) if res_list else 0.0
        avg_qp_lat = np.mean([r.get("mean_qp_latency_ms", 0.0) for r in res_list]) if res_list else 0.0
        avg_total_lat = np.mean([r.get("mean_total_latency_ms", 0.0) for r in res_list]) if res_list else 0.0
        
        succ_rate = (successes / n) * 100.0
        coll_rate = (collisions / n) * 100.0
        oob_rate = (oob / n) * 100.0
        
        succ_ci = wilson_score_interval(successes, n)
        coll_ci = wilson_score_interval(collisions, n)
        
        summary[method] = {
            "num_episodes": n,
            "success_rate_pct": succ_rate,
            "success_ci_95": succ_ci,
            "collision_rate_pct": coll_rate,
            "collision_ci_95": coll_ci,
            "oob_rate_pct": oob_rate,
            "avg_speed_mps": float(avg_speed),
            "avg_completion_time_s": float(avg_time),
            "avg_planner_latency_ms": float(avg_planner_lat),
            "avg_qp_latency_ms": float(avg_qp_lat),
            "avg_total_latency_ms": float(avg_total_lat),
            "avg_latency_ms": float(avg_total_lat),
        }
        
        succ_str = f"{succ_rate:.1f}% [{succ_ci[0]:.1f}, {succ_ci[1]:.1f}]"
        coll_str = f"{coll_rate:.1f}% [{coll_ci[0]:.1f}, {coll_ci[1]:.1f}]"
        print(f"{method:<22} | {succ_str:<20} | {coll_str:<20} | {oob_rate:>4.1f}% | {avg_speed:>10.2f}  | {avg_time:>6.2f}s | {avg_qp_lat:>6.2f}ms | {avg_total_lat:>7.2f}ms")
        
    print(f"{'='*120}\n")
    
    # Save clean unmerged JSON results with metadata
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, "benchmark_comparison.json")
    metadata = {
        "git_commit": get_git_commit(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes_per_method": len(seeds),
        "max_steps": max_steps,
        "num_obstacles": 10,
        "r_margin": 0.38,
        "alpha_w": 2.5,
        "device": str(device)
    }
    payload = {
        "metadata": metadata,
        "summary": summary,
        "episodes": all_results
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Detailed benchmark results saved cleanly to {out_file}")
    
    return summary, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes per baseline (default: 50)")
    parser.add_argument("--max_steps", type=int, default=750, help="Maximum simulation steps per episode (default: 750 = 15.0s)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--methods", nargs="+", default=["mppi_alone", "analytical_hocbf", "neural_icbf", "agile_nicbf_coupled"], help="Methods to benchmark")
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()
    
    run_benchmark(num_episodes=args.episodes, max_steps=args.max_steps, methods=args.methods, device=args.device, save_dir=args.save_dir)


