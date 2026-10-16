"""
run_efg_diagnostics.py -- Diagnostic Suite for Parts E, F, G:
Cell: mu=1.0, v_target=6.0, n=30, same seeds.

Part E:
Per-step trigger logging at K=50 for analytical_hocbf, neural_icbf, agile_nicbf_coupled:
- wall_top_viol (top_margin<0), wall_bot_viol (bot_margin<0), obstacle_viol (any rhs<0), brake_needed
- fraction of steps for each
- fraction of QP-active steps where ONLY violated rows are wall rows
- fraction of QP-active steps where sign(delta_safe) != sign(delta_nom) and |delta_safe - delta_nom| > 0.3
- For neural_icbf and agile_nicbf_coupled:
  * mean min_h_val over steps where vx < 1.0 m/s
  * distance from car to nearest obstacle at first step vx < 1.0 m/s

Part F:
Ablations on analytical_hocbf (K=50):
1. wall rows disabled (diagnostic only; OOB still counted by the env)
2. alpha_w in {2.5, 5.0, 8.0}
3. K in {50, 200, 1000} for planner + shield together

Part G:
Same ablation 1 (wall rows disabled) and K=1000 for neural_icbf and agile_nicbf_coupled:
- neural_icbf no wall
- neural_icbf K=1000
- agile_nicbf_coupled no wall
- agile_nicbf_coupled K=1000
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
for _d in (current_dir, os.path.join(current_dir, "src")):
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

from stress_grid import run_episode, analyze, load, write_meta, METHODS

SEEDS = [101 + i * 37 for i in range(30)]
SHARD = (0, 1)  # (i, n): run seed indices with idx % n == i; set by --shard

def run_suite(methods, K, shield_kwargs, out_file, device):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    done = {(r["method"], r["seed"]) for r in load([out_file])}
    
    # write metadata if not exists
    class DummyArgs:
        def __init__(self):
            self.methods = methods
            self.K = K
            self.shield_kwargs = shield_kwargs
            self.mu = [1.0]
            self.speeds = [6.0]
            self.episodes = len(SEEDS)
    if not os.path.exists(out_file + ".meta.json"):
        write_meta(out_file, DummyArgs())

    with open(out_file, "a") as f:
        for idx, seed in enumerate(SEEDS):
            if idx % SHARD[1] != SHARD[0]:
                continue
            for m in methods:
                if (m, seed) in done:
                    continue
                t0 = time.time()
                rec = run_episode(m, seed, 1.0, 6.0, K, device,
                                  shield_kwargs=shield_kwargs)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[{m:<20} seed={seed}] {rec['outcome']:<12} {rec['time_s']:5.1f}s v={rec['mean_speed']:.2f} ({time.time()-t0:.1f}s)")
    if SHARD[1] > 1:
        return None  # partial results; rerun without --shard to analyze
    return load([out_file])

def run_part_e(device):
    print("\n" + "="*80)
    print("PART E: PER-STEP TRIGGER LOGGING AT K=50 (mu=1.0, v=6.0, n=30)")
    print("="*80)
    methods = ["analytical_hocbf", "neural_icbf", "agile_nicbf_coupled"]
    out_file = "results/part_e_trigger_logging.jsonl"
    recs = run_suite(methods, 50, {"r_margin": 0.38}, out_file, device)
    if recs is None:
        return

    # Print comparative trigger forensics table
    print("\n" + "="*96)
    print(f"{'TRIGGER & BEHAVIORAL FORENSICS (K=50, mu=1.0, v=6.0, n=30)':^96}")
    print("="*96)
    header = f"{'Metric':<54} {'analytical_hocbf':>13} {'neural_icbf':>13} {'agile_nicbf':>13}"
    print(header)
    print("-" * len(header))

    data_by_m = {m: [r for r in recs if r.get("method") == m] for m in methods}

    def fmt_pct(val):
        return f"{val*100:12.1f}%" if val is not None else f"{'--':>13}"
    def fmt_flt(val, prec=3):
        return f"{val:13.{prec}f}" if val is not None else f"{'--':>13}"

    # Fractions over total steps
    for label, key in [
        ("Wall top violation fraction (top_margin < 0)", "frac_wall_top_viol"),
        ("Wall bot violation fraction (bot_margin < 0)", "frac_wall_bot_viol"),
        ("Obstacle violation fraction (any rhs < 0)", "frac_obstacle_viol"),
        ("Brake needed fraction", "frac_brake_needed"),
    ]:
        vals = [np.mean([r[key] for r in data_by_m[m]]) for m in methods]
        print(f"{label:<54} {fmt_pct(vals[0])} {fmt_pct(vals[1])} {fmt_pct(vals[2])}")

    # Fractions over QP-active steps
    for label, key in [
        ("QP-active: ONLY wall rows violated fraction", "frac_qp_active_wall_only"),
        ("QP-active: steering reversal (sign!=sign & |diff|>0.3)", "frac_qp_active_steer_rev"),
    ]:
        vals = [np.mean([r[key] for r in data_by_m[m]]) for m in methods]
        print(f"{label:<54} {fmt_pct(vals[0])} {fmt_pct(vals[1])} {fmt_pct(vals[2])}")

    # Neural specific metrics
    # mean min_h_val over steps where vx < 1.0 m/s
    h_vals = []
    for m in methods:
        h_list = [r["mean_min_h_vx_lt1"] for r in data_by_m[m] if r.get("mean_min_h_vx_lt1") is not None]
        h_vals.append(float(np.mean(h_list)) if h_list else None)
    print(f"{'Mean min_h_val over steps where vx < 1.0 m/s':<54} {fmt_flt(h_vals[0])} {fmt_flt(h_vals[1])} {fmt_flt(h_vals[2])}")

    # distance from car to nearest obstacle at first step vx < 1.0
    dist_vals = []
    for m in methods:
        d_list = [r["dist_obs_first_vx_lt1"] for r in data_by_m[m] if r.get("dist_obs_first_vx_lt1") is not None]
        dist_vals.append(float(np.mean(d_list)) if d_list else None)
    print(f"{'Dist to nearest obs at first step vx < 1.0 (m)':<54} {fmt_flt(dist_vals[0], 2)} {fmt_flt(dist_vals[1], 2)} {fmt_flt(dist_vals[2], 2)}")
    print("="*96 + "\n")

    # Also print standard analyze table
    if recs is not None:
        analyze(recs, ref="analytical_hocbf")

def run_part_f(device):
    print("\n" + "="*80)
    print("PART F: ABLATIONS ON analytical_hocbf (K=50, mu=1.0, v=6.0, n=30)")
    print("="*80)
    
    # 1. wall rows disabled (K=50)
    print("\n--- Ablation F.1: Wall rows disabled (K=50) ---")
    recs_f1 = run_suite(["analytical_hocbf"], 50, {"disable_wall_cbf": True, "r_margin": 0.38},
                        "results/ablation_hocbf_nowall.jsonl", device)
    if recs_f1 is not None:
        analyze(recs_f1, ref="analytical_hocbf")

    # 2. alpha_w in {2.5, 5.0, 8.0} (K=50)
    for aw in [2.5, 5.0, 8.0]:
        print(f"\n--- Ablation F.2: alpha_w = {aw:.1f} (K=50) ---")
        out_f = f"results/ablation_hocbf_aw{int(aw*10):02d}.jsonl"
        recs_aw = run_suite(["analytical_hocbf"], 50, {"alpha_w": aw, "r_margin": 0.38}, out_f, device)
        if recs_aw is not None:
            analyze(recs_aw, ref="analytical_hocbf")

    # 3. K in {50, 200, 1000} (planner + shield together)
    for K_val in [50, 200, 1000]:
        print(f"\n--- Ablation F.3: K = {K_val} (planner + shield) ---")
        out_f = f"results/ablation_hocbf_k{K_val}.jsonl"
        recs_k = run_suite(["analytical_hocbf"], K_val, {"r_margin": 0.38}, out_f, device)
        if recs_k is not None:
            analyze(recs_k, ref="analytical_hocbf")

def run_part_g(device):
    print("\n" + "="*80)
    print("PART G: ABLATION 1 (WALL ROWS DISABLED) & K=1000 FOR NEURAL METHODS")
    print("="*80)

    # 1. neural_icbf with wall rows disabled (K=50)
    print("\n--- G.1: neural_icbf (wall rows disabled, K=50) ---")
    recs_n_nowall = run_suite(["neural_icbf"], 50, {"disable_wall_cbf": True, "r_margin": 0.38},
                              "results/ablation_neural_nowall.jsonl", device)
    if recs_n_nowall is not None:
        analyze(recs_n_nowall, ref="neural_icbf")

    # 2. neural_icbf K=1000 (standard wall rows)
    print("\n--- G.2: neural_icbf (K=1000) ---")
    recs_n_k1000 = run_suite(["neural_icbf"], 1000, {"r_margin": 0.38},
                             "results/ablation_neural_k1000.jsonl", device)
    if recs_n_k1000 is not None:
        analyze(recs_n_k1000, ref="neural_icbf")

    # 3. agile_nicbf_coupled with wall rows disabled (K=50)
    print("\n--- G.3: agile_nicbf_coupled (wall rows disabled, K=50) ---")
    recs_c_nowall = run_suite(["agile_nicbf_coupled"], 50, {"disable_wall_cbf": True, "r_margin": 0.38},
                              "results/ablation_coupled_nowall.jsonl", device)
    if recs_c_nowall is not None:
        analyze(recs_c_nowall, ref="agile_nicbf_coupled")

    # 4. agile_nicbf_coupled K=1000 (standard wall rows)
    print("\n--- G.4: agile_nicbf_coupled (K=1000) ---")
    recs_c_k1000 = run_suite(["agile_nicbf_coupled"], 1000, {"r_margin": 0.38},
                             "results/ablation_coupled_k1000.jsonl", device)
    if recs_c_k1000 is not None:
        analyze(recs_c_k1000, ref="agile_nicbf_coupled")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["E", "F", "G", "all"], default="all")
    ap.add_argument("--shard", default="0/1", help="i/n: run seed indices with idx %% n == i (skips analysis)")
    args = ap.parse_args()
    SHARD = tuple(int(x) for x in args.shard.split("/"))

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.part in ("E", "all"):
        run_part_e(dev)
    if args.part in ("F", "all"):
        run_part_f(dev)
    if args.part in ("G", "all"):
        run_part_g(dev)
