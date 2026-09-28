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
from mpcc_controller import MPCCController
from sfc_planner import SFCMincoPlanner
from drl_racing_policy import DRLRacingPolicy


def run_single_episode(method, seed, max_steps=600, dt=0.02, v_target=6.0, device="cpu"):
    """
    Runs a single closed-loop navigation episode on ClutteredRacingEnv.
    Supported methods:
    1. 'mppi_alone'       (Williams et al., IEEE T-RO 2017)
    2. 'analytical_hocbf' (Xiao & Belta, IEEE TAC 2021)
    3. 'mpcc_liniger'     (Liniger et al., IEEE TCST 2015)
    4. 'sfc_minco'        (Wang & Gao, IEEE T-RO 2022)
    5. 'drl_policy'       (Wurman et al., Nature 2022)
    6. 'neural_icbf'      (Agile-NICBF - Ours, Chandra et al. L4DC 2026)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = ClutteredRacingEnv(num_obstacles=20, track_length=50.0, track_width=4.0, seed=seed)
    state = env.reset(start_vx=3.0)
    
    # Initialize appropriate controller / planner
    mppi_planner = None
    shield = None
    mpcc_controller = None
    sfc_planner = None
    drl_policy = None
    
    if method in ['mppi_alone', 'analytical_hocbf', 'neural_icbf']:
        num_samples = 3000 if (device == "cuda" or "cuda" in str(device)) and torch.cuda.is_available() else 500
        mppi_planner = MPPIPlanner(num_samples=num_samples, horizon=35, v_target=v_target, device=device)
        
    if method in ['analytical_hocbf', 'neural_icbf']:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "neural_icbf_6d.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "neural_icbf_6d.pth")
        shield = NeuralICBFController(model_path=model_path, mode=method, r_margin=0.26, alpha1=2.5, alpha2=3.5, alpha_w=3.5, device=device)
        
    elif method == 'mpcc_liniger':
        mpcc_controller = MPCCController(horizon=20, dt=dt, v_target=v_target, device=device)
        
    elif method == 'sfc_minco':
        sfc_planner = SFCMincoPlanner(horizon_dist=12.0, num_corridors=5, v_target=v_target)
        
    elif method == 'drl_policy':
        drl_policy = DRLRacingPolicy(v_target=v_target, device=device)

    current_u = np.array([2.0, 0.0], dtype=np.float32)
    step_count = 0
    speeds = []
    latencies = []
    interventions = 0
    done = False
    outcome = "timeout"
    
    start_time = time.time()
    
    while not done and step_count < max_steps:
        px, py = state[0], state[1]
        
        # Window nearby obstacles
        nearby_obs = []
        for ox, oy, r_obs in env.obstacles:
            if -2.0 <= (ox - px) <= 15.0 and abs(oy - py) <= 3.0:
                nearby_obs.append([ox, oy, r_obs])
        nearby_obs = np.array(nearby_obs, dtype=np.float32) if len(nearby_obs) > 0 else np.zeros((0, 3), dtype=np.float32)
        
        # 1. Compute Control Action per Method
        t0 = time.time()
        
        if method == 'mppi_alone':
            u_exec = mppi_planner.plan(state, nearby_obs)
            latencies.append((time.time() - t0) * 1000.0)
            
        elif method in ['analytical_hocbf', 'neural_icbf']:
            u_nom = mppi_planner.plan(state, nearby_obs)
            t_qp = time.time()
            u_safe, qp_info = shield.filter_control(state, current_u, u_nom, nearby_obs, dt=dt)
            latencies.append((time.time() - t_qp) * 1000.0)
            if qp_info["qp_active"]:
                interventions += 1
            u_exec = u_safe
            
        elif method == 'mpcc_liniger':
            u_exec, info_m = mpcc_controller.solve(state, nearby_obs)
            latencies.append(info_m["latency_ms"])
            
        elif method == 'sfc_minco':
            u_exec, info_s = sfc_planner.plan(state, nearby_obs)
            latencies.append(info_s["latency_ms"])
            
        elif method == 'drl_policy':
            u_exec, info_d = drl_policy.plan(state, nearby_obs)
            latencies.append(info_d["latency_ms"])
            
        current_u = u_exec
        state, info = env.step(u_exec, dt=dt)
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
            
    total_time = time.time() - start_time
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
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "final_x": float(state[0]),
        "final_y": float(state[1]),
    }


def run_benchmark(num_episodes=10, methods=None, seeds=None, device="cpu", save_dir="./results"):
    """
    Executes comparative evaluation across all 6 methods over specified seeds.
    """
    if methods is None:
        methods = ["mppi_alone", "analytical_hocbf", "mpcc_liniger", "sfc_minco", "drl_policy", "neural_icbf"]
    if seeds is None:
        seeds = [101 + i * 37 for i in range(num_episodes)]
        
    print(f"\n{'='*80}")
    print(f"  6-Way State-of-the-Art Autonomous Racing Benchmark Suite")
    print(f"  Episodes: {len(seeds)} per method | Device: {device}")
    print(f"  Methods: {', '.join(methods)}")
    print(f"{'='*80}\n")
    
    all_results = {m: [] for m in methods}
    
    for method in methods:
        print(f"\n--- Evaluating Baseline: {method} ({len(seeds)} episodes) ---")
        for i, seed in enumerate(seeds):
            res = run_single_episode(method, seed, device=device)
            all_results[method].append(res)
            print(f"  [{i+1:03d}/{len(seeds):03d}] Seed {seed:4d} -> Outcome: {res['outcome']:13s} | Steps: {res['steps']:3d} ({res['traj_time_s']:.2f}s) | Speed: {res['mean_speed']:.2f} m/s | Int: {res['interventions']:2d}")
            
    # Compute Aggregated Metrics
    summary = {}
    print(f"\n\n{'='*95}")
    print(f"{'Method':<22} | {'Success':<9} | {'Collision':<10} | {'OOB':<6} | {'Speed (m/s)':<12} | {'Time (s)':<9} | {'Latency':<10}")
    print(f"{'-'*95}")
    
    for method in methods:
        res_list = all_results[method]
        n = len(res_list)
        successes = sum(1 for r in res_list if r["success"])
        collisions = sum(1 for r in res_list if r["collision"])
        oob = sum(1 for r in res_list if r["out_of_bounds"])
        
        succ_res = [r for r in res_list if r["success"]]
        avg_speed = np.mean([r["mean_speed"] for r in res_list]) if res_list else 0.0
        avg_time = np.mean([r["traj_time_s"] for r in succ_res]) if succ_res else 0.0
        avg_lat = np.mean([r["mean_latency_ms"] for r in res_list if r["mean_latency_ms"] > 0]) if res_list else 0.0
        
        succ_rate = (successes / n) * 100.0
        coll_rate = (collisions / n) * 100.0
        oob_rate = (oob / n) * 100.0
        
        summary[method] = {
            "num_episodes": n,
            "success_rate_pct": succ_rate,
            "collision_rate_pct": coll_rate,
            "oob_rate_pct": oob_rate,
            "avg_speed_mps": float(avg_speed),
            "avg_completion_time_s": float(avg_time),
            "avg_latency_ms": float(avg_lat),
        }
        
        print(f"{method:<22} | {succ_rate:>6.1f}%   | {coll_rate:>7.1f}%    | {oob_rate:>4.1f}% | {avg_speed:>10.2f}   | {avg_time:>7.2f}s  | {avg_lat:>6.2f} ms")
        
    print(f"{'='*95}\n")
    
    # Save JSON results
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, "benchmark_comparison.json")
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "episodes": all_results}, f, indent=2)
    print(f"Detailed benchmark results saved to {out_file}")
    
    return summary, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per baseline")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()
    
    run_benchmark(num_episodes=args.episodes, device=args.device, save_dir=args.save_dir)
