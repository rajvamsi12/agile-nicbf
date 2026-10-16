"""
run_diagnostics.py -- Diagnostic Suite for:
A) Forensics for analytical_hocbf (mu=1.0, v=6.0, K=50, n=30, r_margin=0.38)
B) r_margin sweep for analytical_hocbf (0.21, 0.26, 0.30, 0.38)
C) Feasibility ceiling: mppi_alone (K=1000, n=30)
D) Time series for 3 stalled seeds (r_margin=0.38) and the 10 steps before vx < 1 m/s
"""
import os
import sys
import json
import time
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
for _d in (current_dir, os.path.join(current_dir, "src")):
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

from dynamic_bicycle_env import ClutteredRacingEnv
from mppi_planner import MPPIPlanner
from safe_controller import NeuralICBFController
from stress_grid import run_episode, analyze, load, wilson, get_shield, get_planner

def run_part_a(device="cuda" if torch.cuda.is_available() else "cpu"):
    print("\n" + "="*80)
    print("PART A: FORENSICS (mu=1.0, v=6.0, K=50, n=30, analytical_hocbf, r_margin=0.38)")
    print("="*80)
    out_file = "results/forensics_analytical_hocbf.jsonl"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    seeds = [101 + i * 37 for i in range(30)]
    done = {r["seed"] for r in load([out_file]) if r.get("method") == "analytical_hocbf"}
    
    with open(out_file, "a") as f:
        for i, s in enumerate(seeds):
            if s in done:
                continue
            t0 = time.time()
            rec = run_episode("analytical_hocbf", s, 1.0, 6.0, 50, device,
                              shield_kwargs={"r_margin": 0.38})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{i+1}/30] seed={s} {rec['outcome']:<12} {rec['time_s']:5.1f}s v={rec['mean_speed']:.2f} ({time.time()-t0:.1f}s)")
    
    recs = [r for r in load([out_file]) if r.get("method") == "analytical_hocbf" and r.get("mu") == 1.0 and r.get("v_target") == 6.0]
    analyze(recs, ref="analytical_hocbf")
    return recs

def run_part_b(recs_038, device="cuda" if torch.cuda.is_available() else "cpu"):
    print("\n" + "="*80)
    print("PART B: r_margin SWEEP for analytical_hocbf (0.21, 0.26, 0.30, 0.38)")
    print("="*80)
    margins = [0.21, 0.26, 0.30, 0.38]
    seeds = [101 + i * 37 for i in range(30)]
    all_sweep_recs = []

    for r_m in margins:
        out_file = f"results/sweep_rmargin_{int(round(r_m*100)):03d}.jsonl"
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if r_m == 0.38 and recs_038:
            with open(out_file, "w") as f:
                for r in recs_038:
                    f.write(json.dumps(r) + "\n")
        done = {r["seed"] for r in load([out_file]) if r.get("method") == "analytical_hocbf"}
        with open(out_file, "a") as f:
            for i, s in enumerate(seeds):
                if s in done:
                    continue
                t0 = time.time()
                rec = run_episode("analytical_hocbf", s, 1.0, 6.0, 50, device,
                                  shield_kwargs={"r_margin": r_m})
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[r_m={r_m:.2f} {i+1}/30] seed={s} {rec['outcome']:<12} {rec['time_s']:5.1f}s v={rec['mean_speed']:.2f} ({time.time()-t0:.1f}s)")
        
        recs = [r for r in load([out_file]) if r.get("method") == "analytical_hocbf" and r.get("mu") == 1.0 and r.get("v_target") == 6.0]
        n = len(recs)
        succ = sum(1 for e in recs if e["outcome"] == "success")
        unsafe = sum(1 for e in recs if e["outcome"] in ("collision", "out_of_bounds"))
        stall = sum(1 for e in recs if e["outcome"] == "stalled")
        tmo = sum(1 for e in recs if e["outcome"] == "timeout")
        s_ci = wilson(succ, n)
        u_ci = wilson(unsafe, n)
        mean_v = float(np.mean([e["mean_speed"] for e in recs]))
        mean_intv = float(np.mean([e["intervention_rate"] for e in recs]))
        all_sweep_recs.append({
            "r_margin": r_m, "n": n, "succ": 100*succ/n, "s_ci": s_ci,
            "unsafe": 100*unsafe/n, "u_ci": u_ci, "stall": 100*stall/n, "tmo": 100*tmo/n,
            "mean_v": mean_v, "intv": mean_intv
        })

    print("\n=== r_margin Sweep Summary (analytical_hocbf, mu=1.0, v=6.0, K=50, n=30) ===")
    for row in all_sweep_recs:
        print(f"r_margin={row['r_margin']:.2f}  n={row['n']:<2} | succ {row['succ']:5.1f} [{row['s_ci'][0]:4.1f},{row['s_ci'][1]:5.1f}] | "
              f"unsafe {row['unsafe']:5.1f} [{row['u_ci'][0]:4.1f},{row['u_ci'][1]:5.1f}] | "
              f"stall {row['stall']:5.1f} tmo {row['tmo']:4.1f} | v {row['mean_v']:4.2f} | intv {row['intv']:.2f}")

def run_part_c(device="cuda" if torch.cuda.is_available() else "cpu"):
    print("\n" + "="*80)
    print("PART C: FEASIBILITY CEILING (mppi_alone, K=1000, mu=1.0, v=6.0, n=30)")
    print("="*80)
    out_file = "results/ceiling_mppi_k1000.jsonl"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    seeds = [101 + i * 37 for i in range(30)]
    done = {r["seed"] for r in load([out_file]) if r.get("method") == "mppi_alone"}
    with open(out_file, "a") as f:
        for i, s in enumerate(seeds):
            if s in done:
                continue
            t0 = time.time()
            rec = run_episode("mppi_alone", s, 1.0, 6.0, 1000, device)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[K=1000 {i+1}/30] seed={s} {rec['outcome']:<12} {rec['time_s']:5.1f}s v={rec['mean_speed']:.2f} ({time.time()-t0:.1f}s)")
    
    recs = [r for r in load([out_file]) if r.get("method") == "mppi_alone" and r.get("K") == 1000]
    analyze(recs, ref="mppi_alone")

def run_part_d(stalled_seeds, device="cuda" if torch.cuda.is_available() else "cpu"):
    print("\n" + "="*80)
    print(f"PART D: TIME SERIES FOR 3 STALLED SEEDS {stalled_seeds} (analytical_hocbf, r_margin=0.38)")
    print("="*80)
    dt = 0.02
    for seed in stalled_seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = ClutteredRacingEnv(num_obstacles=10, track_length=50.0, track_width=4.0, seed=seed)
        env.model.p.mu = 1.0
        state = env.reset(start_vx=3.0)
        planner = get_planner(50, 6.0, device)
        shield = get_shield("analytical_hocbf", device, {"r_margin": 0.38})
        current_u = np.array([env.actual_ax, env.actual_delta], dtype=np.float32)

        ts = []
        for step in range(1, 2000):
            perceived = env.get_perceived_obstacles()
            obs_state = env.get_noisy_state()
            u_nom = planner.plan(obs_state, perceived)
            u_exec, qi = shield.filter_control(obs_state, current_u, u_nom, perceived, dt=dt)
            ts.append({
                "step": step,
                "t": round(step * dt, 3),
                "x": float(state[0]),
                "y": float(state[1]),
                "vx": float(state[3]),
                "ax_nom": float(u_nom[0]),
                "ax_safe": float(u_exec[0]),
                "delta_nom": float(u_nom[1]),
                "delta_safe": float(u_exec[1]),
                "qp_status": str(qi.get("qp_status", "")),
                "wall_fallback": bool(qi.get("wall_fallback", False)),
            })
            state, info = env.step(u_exec, dt=dt)
            current_u = info["actual_u"]
            if info["collision"] or info["out_of_bounds"] or info["success"]:
                break
            if state[3] < 0.5 and step > 150: # stalled check
                pass

        ts_file = f"results/timeseries_stalled_seed_{seed}.json"
        with open(ts_file, "w") as f:
            json.dump(ts, f, indent=2)
        print(f"\nSaved {len(ts)} steps to {ts_file}")

        # Find first step where vx < 1.0
        idx_lt1 = None
        for idx, row in enumerate(ts):
            if row["vx"] < 1.0:
                idx_lt1 = idx
                break

        print(f"\n--- Stalled Seed {seed}: 10 steps just before vx drops below 1.0 m/s (first vx<1.0 at step {ts[idx_lt1]['step'] if idx_lt1 is not None else 'N/A'}, vx={ts[idx_lt1]['vx']:.2f} m/s) ---")
        if idx_lt1 is not None and idx_lt1 >= 10:
            sub = ts[idx_lt1-10:idx_lt1]
        elif idx_lt1 is not None:
            sub = ts[:idx_lt1]
        else:
            sub = ts[-10:]

        header = f"{'Step':<5} {'t(s)':<6} {'x(m)':<7} {'y(m)':<7} {'vx(m/s)':<8} {'ax_nom':<8} {'ax_safe':<8} {'d_nom':<8} {'d_safe':<8} {'qp_status':<22} {'wall_fb':<7}"
        print(header)
        print("-" * len(header))
        for r in sub:
            print(f"{r['step']:<5} {r['t']:<6.2f} {r['x']:<7.2f} {r['y']:<7.2f} {r['vx']:<8.2f} {r['ax_nom']:<8.2f} {r['ax_safe']:<8.2f} {r['delta_nom']:<8.3f} {r['delta_safe']:<8.3f} {r['qp_status']:<22} {str(r['wall_fallback']):<7}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "C", "D", "all"], default="all")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recs_a = None

    if args.part in ("A", "all"):
        recs_a = run_part_a(dev)
    if args.part in ("B", "all"):
        if recs_a is None:
            out_file = "results/forensics_analytical_hocbf.jsonl"
            recs_a = [r for r in load([out_file]) if r.get("method") == "analytical_hocbf" and r.get("mu") == 1.0 and r.get("v_target") == 6.0]
        run_part_b(recs_a, dev)
    if args.part in ("C", "all"):
        run_part_c(dev)
    if args.part in ("D", "all"):
        if recs_a is None:
            out_file = "results/forensics_analytical_hocbf.jsonl"
            recs_a = [r for r in load([out_file]) if r.get("method") == "analytical_hocbf" and r.get("mu") == 1.0 and r.get("v_target") == 6.0]
        stalled = [r["seed"] for r in recs_a if r.get("outcome") == "stalled"][:3]
        if len(stalled) < 3:
            stalled = [101, 138, 286]
        run_part_d(stalled, dev)
