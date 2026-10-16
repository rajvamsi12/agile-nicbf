"""
stress_grid.py -- paired stress-test harness for MPPI vs. shielded variants.

Grid: plant friction mu x target speed x MPPI sample count K.
For a given seed, the obstacle layout and the sensor-noise RNG stream are identical
across methods and across grid cells (layout depends on the seed only), so comparisons
are paired.

mu is set on the PLANT only (env.model.p.mu). MPPI and the shield build their own
default models (mu = 1.0489), so mu < 1.0489 is a genuine model mismatch for every method.
The nominal plant mu is 1.0489 (env also jitters it +-5% per seed in __init__; the
override below replaces that jitter with the fixed grid value).

Outcomes: success | collision | out_of_bounds | stalled | timeout
  stalled = vx < 0.5 m/s for more than 3 s (episode ends early).
  timeout = still moving at max_steps (default 2000 steps = 40 s).

Usage:
  python stress_grid.py --episodes 50 --Ks 50                 # 3x3 grid (mu x speed) at K=50
  python stress_grid.py --episodes 50 --Ks 20 50 100          # full 3x3x3
  python stress_grid.py --shard 0/2 --out results/sg_0.jsonl  # parallel processes
  python stress_grid.py --analyze results/sg_0.jsonl results/sg_1.jsonl
  python stress_grid.py --no-brake-clamp --out results/sg_noclamp.jsonl   # needs brake_clamp patch

Resumable: finished (method, seed, mu, speed, K) episodes in --out are skipped.
"""
import os
import sys
import json
import time
import hashlib
import argparse
import subprocess
import numpy as np
import torch
from scipy import stats

current_dir = os.path.dirname(os.path.abspath(__file__))
for _d in (current_dir, os.path.join(current_dir, "src")):
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

from dynamic_bicycle_env import ClutteredRacingEnv
from mppi_planner import MPPIPlanner
from safe_controller import NeuralICBFController

METHODS = ["mppi_alone", "mppi_oracle_mu", "analytical_hocbf", "neural_icbf", "agile_nicbf_coupled"]
STALL_SPEED = 0.5      # m/s
STALL_STEPS = 150      # 3.0 s at dt = 0.02
UNSAFE = ("collision", "out_of_bounds")

_SHIELDS, _PLANNERS = {}, {}


def find_model():
    for p in (os.path.join(current_dir, "models", "neural_icbf_6d.pth"),
              os.path.join(current_dir, "..", "models", "neural_icbf_6d.pth")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("neural_icbf_6d.pth not found next to or above this script")


def get_shield(mode, device, kwargs):
    key = (mode, device, tuple(sorted(kwargs.items())))
    if key not in _SHIELDS:
        kw = dict(kwargs)
        r_margin = kw.pop("r_margin", 0.38)
        _SHIELDS[key] = NeuralICBFController(
            model_path=find_model(), mode=mode, r_margin=r_margin,
            alpha1=2.5, alpha2=3.5, alpha_w=kw.pop("alpha_w", 2.5), device=device, **kw)
    return _SHIELDS[key]


def get_planner(K, v_target, device):
    key = (K, v_target, device)
    if key not in _PLANNERS:
        _PLANNERS[key] = MPPIPlanner(num_samples=K, horizon=30, v_target=v_target,
                                     r_margin=0.38, device=device)
    _PLANNERS[key].reset()
    return _PLANNERS[key]


def run_episode(method, seed, mu, v_target, K, device, max_steps=2000, dt=0.02, shield_kwargs=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = ClutteredRacingEnv(num_obstacles=10, track_length=50.0, track_width=4.0, seed=seed)
    env.model.p.mu = float(mu)               # plant only -> model mismatch for MPPI + shield
    state = env.reset(start_vx=3.0)

    if method == "mppi_oracle_mu":
        planner = MPPIPlanner(num_samples=K, horizon=30, v_target=v_target,
                              r_margin=0.38, device=device)
        planner.model.p.mu = float(mu)
        planner.p.mu = float(mu)
    else:
        planner = get_planner(K, v_target, device)

    shield = None
    if method in ("nicbf", "nicbf_mppi"):
        _dyn = os.path.join(current_dir, "prototype_neural_ode", "dynamic")
        if _dyn not in sys.path:
            sys.path.insert(0, _dyn)
        from nicbf_controller import NeuralICBF
        kw = dict((shield_kwargs or {}).get("nicbf", {}))
        shield = NeuralICBF(nominal="mppi" if method == "nicbf_mppi" else "pi", **kw)
    elif method == "predictive_filter":
        from predictive_filter import PredictiveSafetyFilter
        psf_kw = dict((shield_kwargs or {}).get("psf", {}))
        shield = PredictiveSafetyFilter(device=psf_kw.pop("device", "cpu"), **psf_kw)  # CPU: 17 ms vs 42 ms on GPU
    elif method not in ("mppi_alone", "mppi_oracle_mu"):
        mode = "neural_icbf" if method in ("neural_icbf", "agile_nicbf_coupled") else "analytical_hocbf"
        shield = get_shield(mode, device, shield_kwargs or {})

    current_u = np.array([env.actual_ax, env.actual_delta], dtype=np.float32)
    speeds, interventions, brakes, stall_run = [], 0, 0, 0
    n_surv_list = []
    shield_infos = []
    outcome, step = "timeout", 0
    dist_obs_first_vx_lt1 = None

    for step in range(1, max_steps + 1):
        perceived = env.get_perceived_obstacles()
        obs_state = env.get_noisy_state()

        if method in ("mppi_alone", "mppi_oracle_mu"):
            u_exec = planner.plan(obs_state, perceived)
        elif method in ("nicbf", "nicbf_mppi"):
            u_nom = planner.plan(obs_state, perceived) if method == "nicbf_mppi" else None
            u_exec, qi = shield.step(obs_state, current_u, perceived, v_target, u_nom=u_nom, dt=dt)
            interventions += int(qi["qp_active"])
            shield_infos.append({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                                 for k, v in qi.items() if k != "u_target"})
        else:
            net = shield.barrier_net if method == "agile_nicbf_coupled" else None
            u_nom = planner.plan(obs_state, perceived, barrier_net=net)
            if method == "predictive_filter":   # planner.U is already shifted: it is the plan after u_nom
                u_exec, qi = shield.filter_control(obs_state, current_u, u_nom, perceived, dt=dt,
                                                   plan_tail=planner.U.detach().cpu().numpy(),
                                                   samples=getattr(planner, "last_samples", None),
                                                   sample_costs=getattr(planner, "last_costs", None))
            else:
                u_exec, qi = shield.filter_control(obs_state, current_u, u_nom, perceived, dt=dt)
            interventions += int(qi["qp_active"])
            brakes += int(qi["brake_override"])
            shield_infos.append({
                "qp_status": str(qi.get("qp_status", "unknown")),
                "wall_fallback": bool(qi.get("wall_fallback", False)),
                "n_obstacle_rows": int(qi.get("n_obstacle_rows", 0)),
                "n_unreachable": int(qi.get("n_unreachable", 0)),
                "dev_ax": float(u_exec[0] - u_nom[0]),
                "dev_delta": float(u_exec[1] - u_nom[1]),
                "qp_active": bool(qi.get("qp_active", False)),
                "wall_margin_min": float(qi.get("wall_margin_min", 0.0)),
                "min_h_val": float(qi.get("min_h_val", 1.0)),
                "wall_top_viol": bool(qi.get("wall_top_viol", False)),
                "wall_bot_viol": bool(qi.get("wall_bot_viol", False)),
                "obstacle_viol": bool(qi.get("obstacle_viol", False)),
                "brake_needed": bool(qi.get("brake_needed", False)),
                "wall_steer_inverted": bool(qi.get("wall_steer_inverted", False)),
                "delta_safe": float(u_exec[1]),
                "delta_nom": float(u_nom[1]),
                "vx": float(state[3]),
            })

        n_surv_list.append(getattr(planner, "last_n_surv", 0))
        state, info = env.step(u_exec, dt=dt)
        current_u = info["actual_u"]
        speeds.append(float(state[3]))
        stall_run = stall_run + 1 if state[3] < STALL_SPEED else 0

        if dist_obs_first_vx_lt1 is None and state[3] < 1.0 and len(perceived) > 0:
            dists = [np.hypot(ox - state[0], oy - state[1]) - ro for ox, oy, ro in perceived]
            dist_obs_first_vx_lt1 = float(min(dists))

        if info["collision"]:
            outcome = "collision"; break
        if info["out_of_bounds"]:
            outcome = "out_of_bounds"; break
        if info["success"]:
            outcome = "success"; break
        if stall_run > STALL_STEPS:
            outcome = "stalled"; break

    mean_n_surv = float(np.mean(n_surv_list)) if n_surv_list else 0.0
    frac_steps_n_surv_lt5 = float(np.mean([int(n < 5) for n in n_surv_list])) if n_surv_list else 0.0

    rec = {
        "method": method, "seed": int(seed), "mu": float(mu), "v_target": float(v_target),
        "K": int(K), "outcome": outcome, "steps": step, "time_s": step * dt,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "intervention_rate": interventions / max(1, step),
        "brake_override_rate": brakes / max(1, step),
        "final_x": float(state[0]),
        "mean_n_surv": mean_n_surv,
        "frac_steps_n_surv_lt5": frac_steps_n_surv_lt5,
    }
    if shield_kwargs and "r_margin" in shield_kwargs:
        rec["r_margin"] = float(shield_kwargs["r_margin"])

    if shield_infos and method in ("nicbf", "nicbf_mppi"):
        si = shield_infos
        rec["frac_qp_ok"] = float(np.mean([s["qp_status"] == "ok" for s in si]))
        rec["frac_slack_used"] = float(np.mean([float(s["slack_max"]) > 1e-4 for s in si]))
        rec["mean_min_h_obs"] = float(np.mean([float(s["min_h_obs"]) for s in si]))
        rec["min_min_h_obs"] = float(np.min([float(s["min_h_obs"]) for s in si]))
        rec["min_min_h_wall"] = float(np.min([float(s["min_h_wall"]) for s in si]))
        rec["frac_grip_viol"] = float(np.mean([float(s["h_grip"]) < 0 for s in si]))
        rec["mean_latency_ms"] = float(np.mean([float(s["latency_ms"]) for s in si]))
        rec["p95_latency_ms"] = float(np.percentile([float(s["latency_ms"]) for s in si], 95))
    elif shield_infos:
        n_si = len(shield_infos)
        rec["frac_qp_ok"] = float(sum(1 for s in shield_infos if s["qp_status"] == "ok") / n_si)
        rec["frac_qp_skipped"] = float(sum(1 for s in shield_infos if s["qp_status"] == "skipped_all_satisfied") / n_si)
        rec["frac_qp_infeasible"] = float(sum(1 for s in shield_infos if s["qp_status"] == "infeasible") / n_si)
        rec["frac_qp_exception"] = float(sum(1 for s in shield_infos if s["qp_status"].startswith("exception")) / n_si)
        rec["frac_wall_fallback"] = float(sum(1 for s in shield_infos if s["wall_fallback"]) / n_si)
        rec["mean_n_obstacle_rows"] = float(np.mean([s["n_obstacle_rows"] for s in shield_infos]))
        rec["mean_n_unreachable"] = float(np.mean([s["n_unreachable"] for s in shield_infos]))
        rec["mean_dev_ax"] = float(np.mean([s["dev_ax"] for s in shield_infos]))
        rec["mean_dev_delta"] = float(np.mean([s["dev_delta"] for s in shield_infos]))
        interv_steps = [s for s in shield_infos if s["qp_active"]]
        rec["frac_interv_ax_reduced"] = float(sum(1 for s in interv_steps if s["dev_ax"] < -1e-4) / len(interv_steps)) if interv_steps else 0.0
        rec["mean_wall_margin_min"] = float(np.mean([s["wall_margin_min"] for s in shield_infos]))
        rec["mean_min_h_val"] = float(np.mean([s["min_h_val"] for s in shield_infos]))

        rec["frac_wall_top_viol"] = float(sum(1 for s in shield_infos if s["wall_top_viol"]) / n_si)
        rec["frac_wall_bot_viol"] = float(sum(1 for s in shield_infos if s["wall_bot_viol"]) / n_si)
        rec["frac_obstacle_viol"] = float(sum(1 for s in shield_infos if s["obstacle_viol"]) / n_si)
        rec["frac_brake_needed"] = float(sum(1 for s in shield_infos if s["brake_needed"]) / n_si)

        n_act = len(interv_steps)
        if n_act > 0:
            wall_only = sum(1 for s in interv_steps if (s["wall_top_viol"] or s["wall_bot_viol"]) and not s["obstacle_viol"] and not s["brake_needed"])
            rec["frac_qp_active_wall_only"] = float(wall_only / n_act)
            steer_rev = sum(1 for s in interv_steps if (np.sign(s["delta_safe"]) != np.sign(s["delta_nom"])) and (abs(s["delta_safe"] - s["delta_nom"]) > 0.3))
            rec["frac_qp_active_steer_rev"] = float(steer_rev / n_act)
        else:
            rec["frac_qp_active_wall_only"] = 0.0
            rec["frac_qp_active_steer_rev"] = 0.0

        vx_lt1_steps = [s for s in shield_infos if s["vx"] < 1.0]
        rec["mean_min_h_vx_lt1"] = float(np.mean([s["min_h_val"] for s in vx_lt1_steps])) if vx_lt1_steps else None
        rec["dist_obs_first_vx_lt1"] = dist_obs_first_vx_lt1

    return rec


# ----------------------------------------------------------------------------- stats
def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z / d * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100.0 * max(0.0, c - m), 100.0 * min(1.0, c + m)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p from discordant counts b (ref only) and c (method only)."""
    n = b + c
    return 1.0 if n == 0 else float(stats.binomtest(min(b, c), n, 0.5).pvalue)


def load(paths):
    recs = []
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                recs += [json.loads(line) for line in f if line.strip()]
    return recs


def analyze(recs, ref="mppi_alone", out_json=None):
    cells = {}
    for r in recs:
        cells.setdefault((r["mu"], r["v_target"], r["K"]), {}).setdefault(r["method"], {})[r["seed"]] = r

    summary = {}
    for cell in sorted(cells, key=lambda c: (-c[0], c[1], c[2])):
        mu, v, K = cell
        by_m = cells[cell]
        print(f"\n=== mu={mu:.2f}  v_target={v:.1f} m/s  K={K} ===")
        summary[str(cell)] = {}
        for m in METHODS:
            if m not in by_m:
                continue
            eps = list(by_m[m].values())
            n = len(eps)
            cnt = lambda pred: sum(1 for e in eps if pred(e))
            succ = cnt(lambda e: e["outcome"] == "success")
            unsafe = cnt(lambda e: e["outcome"] in UNSAFE)
            stall = cnt(lambda e: e["outcome"] == "stalled")
            tmo = cnt(lambda e: e["outcome"] == "timeout")
            s_ci, u_ci = wilson(succ, n), wilson(unsafe, n)
            t_succ = [e["time_s"] for e in eps if e["outcome"] == "success"]
            mean_ns = float(np.mean([e.get("mean_n_surv", 0.0) for e in eps]))
            frac_lt5 = float(np.mean([e.get("frac_steps_n_surv_lt5", 0.0) for e in eps]))
            row = {
                "n": n, "success_pct": 100 * succ / n, "success_ci": s_ci,
                "unsafe_pct": 100 * unsafe / n, "unsafe_ci": u_ci,
                "stalled_pct": 100 * stall / n, "timeout_pct": 100 * tmo / n,
                "mean_speed": float(np.mean([e["mean_speed"] for e in eps])),
                "median_finish_s": float(np.median(t_succ)) if t_succ else None,
                "intervention_rate": float(np.mean([e["intervention_rate"] for e in eps])),
                "brake_override_rate": float(np.mean([e["brake_override_rate"] for e in eps])),
                "mean_n_surv": mean_ns,
                "frac_steps_n_surv_lt5": frac_lt5,
            }
            if m != ref and ref in by_m:
                seeds = sorted(set(by_m[m]) & set(by_m[ref]))
                a = [by_m[ref][s]["outcome"] == "success" for s in seeds]
                b_ = [by_m[m][s]["outcome"] == "success" for s in seeds]
                row["p_success_vs_ref"] = mcnemar_exact(sum(x and not y for x, y in zip(a, b_)),
                                                        sum(y and not x for x, y in zip(a, b_)))
                ua = [by_m[ref][s]["outcome"] in UNSAFE for s in seeds]
                ub = [by_m[m][s]["outcome"] in UNSAFE for s in seeds]
                row["p_unsafe_vs_ref"] = mcnemar_exact(sum(x and not y for x, y in zip(ua, ub)),
                                                       sum(y and not x for x, y in zip(ua, ub)))
                both = [s for s in seeds if by_m[ref][s]["outcome"] == "success"
                        and by_m[m][s]["outcome"] == "success"]
                row["n_both_success"] = len(both)
                try:
                    row["p_time_vs_ref"] = float(stats.wilcoxon(
                        [by_m[m][s]["time_s"] for s in both],
                        [by_m[ref][s]["time_s"] for s in both]).pvalue) if len(both) >= 6 else None
                except ValueError:
                    row["p_time_vs_ref"] = None
            summary[str(cell)][m] = row

            ft = f"{row['median_finish_s']:.1f}" if row["median_finish_s"] else " -- "
            line = (f"{m:<22} n={n:<3} succ {row['success_pct']:5.1f} [{s_ci[0]:4.1f},{s_ci[1]:5.1f}] | "
                    f"unsafe {row['unsafe_pct']:5.1f} [{u_ci[0]:4.1f},{u_ci[1]:5.1f}] | "
                    f"stall {row['stalled_pct']:4.1f} tmo {row['timeout_pct']:4.1f} | "
                    f"v {row['mean_speed']:4.2f} T {ft} | intv {row['intervention_rate']:.2f} "
                    f"brk {row['brake_override_rate']:.2f} | "
                    f"n_surv {mean_ns:4.1f} lt5 {frac_lt5*100:4.1f}%")
            print(line)
            if "p_success_vs_ref" in row:
                pt = row.get("p_time_vs_ref")
                print(f"{'':<22} vs {ref}: p_succ={row['p_success_vs_ref']:.3f} "
                      f"p_unsafe={row['p_unsafe_vs_ref']:.3f} "
                      f"p_time={'n/a' if pt is None else f'{pt:.3f}'} (n_both={row['n_both_success']})")

            f_eps = [e for e in eps if "frac_qp_ok" in e]
            if f_eps:
                print(f"\n================================================================================")
                print(f"SHIELD FORENSICS: {m} (cell mu={mu:.2f}, v={v:.1f}, K={K}, n={len(f_eps)})")
                print(f"================================================================================")
                print(f"  {'Metric':<52} {'Value':>22}")
                print(f"  {'-'*76}")
                print(f"  {'QP status: ok fraction':<52} {np.mean([e['frac_qp_ok'] for e in f_eps])*100:>21.1f}%")
                print(f"  {'QP status: skipped_all_satisfied fraction':<52} {np.mean([e['frac_qp_skipped'] for e in f_eps])*100:>21.1f}%")
                print(f"  {'QP status: infeasible fraction':<52} {np.mean([e['frac_qp_infeasible'] for e in f_eps])*100:>21.1f}%")
                print(f"  {'QP status: exception:* fraction':<52} {np.mean([e['frac_qp_exception'] for e in f_eps])*100:>21.1f}%")
                print(f"  {'Wall fallback active fraction':<52} {np.mean([e['frac_wall_fallback'] for e in f_eps])*100:>21.1f}%")
                print(f"  {'Mean obstacle constraint rows (n_obstacle_rows)':<52} {np.mean([e['mean_n_obstacle_rows'] for e in f_eps]):>22.2f}")
                print(f"  {'Mean unreachable obstacles (n_unreachable)':<52} {np.mean([e['mean_n_unreachable'] for e in f_eps]):>22.2f}")
                print(f"  {'Mean (u_safe - u_nom) ax [m/s^2]':<52} {np.mean([e['mean_dev_ax'] for e in f_eps]):>+22.3f}")
                print(f"  {'Mean (u_safe - u_nom) delta [rad]':<52} {np.mean([e['mean_dev_delta'] for e in f_eps]):>+22.3f}")
                print(f"  {'Fraction of interventions with ax reduced (braking)':<52} {np.mean([e['frac_interv_ax_reduced'] for e in f_eps])*100:>21.1f}%")
                print(f"  {'Mean minimum wall margin (wall_margin_min)':<52} {np.mean([e['mean_wall_margin_min'] for e in f_eps]):>+22.3f}")
                print(f"  {'Mean minimum obstacle h(x) (min_h_val)':<52} {np.mean([e['mean_min_h_val'] for e in f_eps]):>+22.3f}")
                print(f"  {'Wall top violation fraction (top_margin < 0)':<52} {np.mean([e.get('frac_wall_top_viol', 0.0) for e in f_eps])*100:>21.1f}%")
                print(f"  {'Wall bot violation fraction (bot_margin < 0)':<52} {np.mean([e.get('frac_wall_bot_viol', 0.0) for e in f_eps])*100:>21.1f}%")
                print(f"  {'Obstacle violation fraction (any rhs < 0)':<52} {np.mean([e.get('frac_obstacle_viol', 0.0) for e in f_eps])*100:>21.1f}%")
                print(f"  {'Brake needed fraction':<52} {np.mean([e.get('frac_brake_needed', 0.0) for e in f_eps])*100:>21.1f}%")
                print(f"  {'QP-active steps: ONLY wall rows violated fraction':<52} {np.mean([e.get('frac_qp_active_wall_only', 0.0) for e in f_eps])*100:>21.1f}%")
                print(f"  {'QP-active steps: steering reversal & |diff|>0.3':<52} {np.mean([e.get('frac_qp_active_steer_rev', 0.0) for e in f_eps])*100:>21.1f}%")
                h_vx_lt1 = [e['mean_min_h_vx_lt1'] for e in f_eps if e.get('mean_min_h_vx_lt1') is not None]
                if h_vx_lt1:
                    print(f"  {'Mean min_h_val over steps with vx < 1.0 m/s':<52} {np.mean(h_vx_lt1):>+22.3f}")
                dist_vx_lt1 = [e['dist_obs_first_vx_lt1'] for e in f_eps if e.get('dist_obs_first_vx_lt1') is not None]
                if dist_vx_lt1:
                    print(f"  {'Mean dist to nearest obs at first step vx < 1.0':<52} {np.mean(dist_vx_lt1):>21.2f} m")
                print(f"================================================================================\n")
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to {out_json}")
    return summary


# ----------------------------------------------------------------------------- main
def file_md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def write_meta(out_path, args):
    srcs = ["dynamic_bicycle_env.py", "mppi_planner.py", "safe_controller.py", "train_neural_icbf.py", "stress_grid.py"]
    hashes = {}
    for s in srcs:
        for d in (current_dir, os.path.join(current_dir, "src")):
            p = os.path.join(d, s)
            if os.path.exists(p):
                hashes[s] = file_md5(p)
                break
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         cwd=current_dir).decode().strip()
    except Exception:
        commit = "unknown"
    meta = {"args": vars(args), "source_md5": hashes, "git_commit": commit,
            "model_md5": file_md5(find_model()), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(out_path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, nargs="+", default=[1.0, 0.7, 0.5])
    ap.add_argument("--speeds", type=float, nargs="+", default=[6.0, 8.0, 10.0])
    ap.add_argument("--Ks", type=int, nargs="+", default=[50])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/stress_grid.jsonl")
    ap.add_argument("--shard", default="0/1", help="i/n: run seed indices with idx %% n == i")
    ap.add_argument("--r_margin", type=float, default=0.38, help="shield safety boundary margin")
    ap.add_argument("--no-brake-clamp", action="store_true",
                    help="pass brake_clamp=False to the shield (requires the brake_clamp patch)")
    ap.add_argument("--wall-steer-guard", default="off", choices=["off", "zero_steer", "drop_row"],
                    help="shield handling of wall rows with inverted steering authority")
    ap.add_argument("--obs-steer-guard", action="store_true", help="zero inverted steering authority in obstacle rows")
    ap.add_argument("--slip-brake-limit", type=float, default=None, help="max extra QP braking (m/s^2) at high slip/yaw")
    ap.add_argument("--nicbf", default=None, help='JSON kwargs for the Neural ICBF pipeline (nicbf, nicbf_mppi)')
    ap.add_argument("--psf", default=None, help='JSON kwargs for predictive_filter, e.g. {"n_samples": 0}')
    ap.add_argument("--analyze", nargs="*", default=None, help="analyze jsonl file(s); default --out")
    ap.add_argument("--ref", default="mppi_alone")
    args = ap.parse_args()

    if args.analyze is not None:
        files = args.analyze or [args.out]
        analyze(load(files), ref=args.ref, out_json=os.path.splitext(files[0])[0] + "_summary.json")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_meta(args.out, args)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    seeds = [101 + i * 37 for i in range(args.episodes)]
    seeds = [s for i, s in enumerate(seeds) if i % shard_n == shard_i]
    shield_kwargs = {"r_margin": args.r_margin}
    if args.no_brake_clamp:
        shield_kwargs["brake_clamp"] = False
    if args.obs_steer_guard:
        shield_kwargs["obs_steer_guard"] = True
    if args.slip_brake_limit is not None:
        shield_kwargs["slip_brake_limit"] = args.slip_brake_limit
    if args.psf:
        shield_kwargs["psf"] = json.loads(args.psf)
    if args.nicbf:
        shield_kwargs["nicbf"] = json.loads(args.nicbf)
    if args.wall_steer_guard != "off":
        shield_kwargs["wall_steer_guard"] = args.wall_steer_guard

    done = {(r["method"], r["seed"], r["mu"], r["v_target"], r["K"]) for r in load([args.out])}
    cells = [(mu, v, K) for mu in args.mu for v in args.speeds for K in args.Ks]
    total = len(cells) * len(seeds) * len(args.methods)
    print(f"{len(cells)} cells x {len(seeds)} seeds x {len(args.methods)} methods = {total} episodes "
          f"({len(done)} already done) | device {args.device}")

    n_run = 0
    with open(args.out, "a") as f:
        for mu, v, K in cells:
            for seed in seeds:
                for m in args.methods:
                    if (m, seed, float(mu), float(v), int(K)) in done:
                        continue
                    t0 = time.time()
                    rec = run_episode(m, seed, mu, v, K, args.device,
                                      max_steps=args.max_steps, shield_kwargs=shield_kwargs)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    n_run += 1
                    print(f"[{n_run}] mu={mu} v={v} K={K} seed={seed} {m:<20} "
                          f"{rec['outcome']:<13} {rec['time_s']:5.1f}s v={rec['mean_speed']:.2f} "
                          f"({time.time() - t0:.1f}s wall)", flush=True)

    analyze(load([args.out]), ref=args.ref, out_json=os.path.splitext(args.out)[0] + "_summary.json")


if __name__ == "__main__":
    main()
