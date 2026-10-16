with open('/home/raj/agile_nicbf/stress_grid.py', 'r') as f:
    text = f.read()

# 1. Update METHODS
old_methods = 'METHODS = ["mppi_alone", "analytical_hocbf", "neural_icbf", "agile_nicbf_coupled"]'
new_methods = 'METHODS = ["mppi_alone", "mppi_oracle_mu", "analytical_hocbf", "neural_icbf", "agile_nicbf_coupled"]'
assert old_methods in text
text = text.replace(old_methods, new_methods)

# 2. Update run_episode
old_run_ep = """    planner = get_planner(K, v_target, device)
    shield = None
    if method != "mppi_alone":
        mode = "neural_icbf" if method in ("neural_icbf", "agile_nicbf_coupled") else "analytical_hocbf"
        shield = get_shield(mode, device, shield_kwargs or {})

    current_u = np.array([env.actual_ax, env.actual_delta], dtype=np.float32)
    speeds, interventions, brakes, stall_run = [], 0, 0, 0
    outcome, step = "timeout", 0

    for step in range(1, max_steps + 1):
        perceived = env.get_perceived_obstacles()
        obs_state = env.get_noisy_state()

        if method == "mppi_alone":
            u_exec = planner.plan(obs_state, perceived)
        else:
            net = shield.barrier_net if method == "agile_nicbf_coupled" else None
            u_nom = planner.plan(obs_state, perceived, barrier_net=net)
            u_exec, qi = shield.filter_control(obs_state, current_u, u_nom, perceived, dt=dt)
            interventions += int(qi["qp_active"])
            brakes += int(qi["brake_override"])

        state, info = env.step(u_exec, dt=dt)
        current_u = info["actual_u"]
        speeds.append(float(state[3]))
        stall_run = stall_run + 1 if state[3] < STALL_SPEED else 0

        if info["collision"]:
            outcome = "collision"; break
        if info["out_of_bounds"]:
            outcome = "out_of_bounds"; break
        if info["success"]:
            outcome = "success"; break
        if stall_run > STALL_STEPS:
            outcome = "stalled"; break

    return {
        "method": method, "seed": int(seed), "mu": float(mu), "v_target": float(v_target),
        "K": int(K), "outcome": outcome, "steps": step, "time_s": step * dt,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "intervention_rate": interventions / max(1, step),
        "brake_override_rate": brakes / max(1, step),
        "final_x": float(state[0]),
    }"""

new_run_ep = """    if method == "mppi_oracle_mu":
        planner = MPPIPlanner(num_samples=K, horizon=30, v_target=v_target,
                              r_margin=0.38, device=device)
        planner.model.p.mu = float(mu)
        planner.p.mu = float(mu)
    else:
        planner = get_planner(K, v_target, device)

    shield = None
    if method not in ("mppi_alone", "mppi_oracle_mu"):
        mode = "neural_icbf" if method in ("neural_icbf", "agile_nicbf_coupled") else "analytical_hocbf"
        shield = get_shield(mode, device, shield_kwargs or {})

    current_u = np.array([env.actual_ax, env.actual_delta], dtype=np.float32)
    speeds, interventions, brakes, stall_run = [], 0, 0, 0
    n_surv_list = []
    outcome, step = "timeout", 0

    for step in range(1, max_steps + 1):
        perceived = env.get_perceived_obstacles()
        obs_state = env.get_noisy_state()

        if method in ("mppi_alone", "mppi_oracle_mu"):
            u_exec = planner.plan(obs_state, perceived)
        else:
            net = shield.barrier_net if method == "agile_nicbf_coupled" else None
            u_nom = planner.plan(obs_state, perceived, barrier_net=net)
            u_exec, qi = shield.filter_control(obs_state, current_u, u_nom, perceived, dt=dt)
            interventions += int(qi["qp_active"])
            brakes += int(qi["brake_override"])

        n_surv_list.append(planner.last_n_surv)
        state, info = env.step(u_exec, dt=dt)
        current_u = info["actual_u"]
        speeds.append(float(state[3]))
        stall_run = stall_run + 1 if state[3] < STALL_SPEED else 0

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

    return {
        "method": method, "seed": int(seed), "mu": float(mu), "v_target": float(v_target),
        "K": int(K), "outcome": outcome, "steps": step, "time_s": step * dt,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "intervention_rate": interventions / max(1, step),
        "brake_override_rate": brakes / max(1, step),
        "final_x": float(state[0]),
        "mean_n_surv": mean_n_surv,
        "frac_steps_n_surv_lt5": frac_steps_n_surv_lt5,
    }"""

assert old_run_ep in text
text = text.replace(old_run_ep, new_run_ep)

# 3. Update analyze
old_analyze_row = """            row = {
                "n": n, "success_pct": 100 * succ / n, "success_ci": s_ci,
                "unsafe_pct": 100 * unsafe / n, "unsafe_ci": u_ci,
                "stalled_pct": 100 * stall / n, "timeout_pct": 100 * tmo / n,
                "mean_speed": float(np.mean([e["mean_speed"] for e in eps])),
                "median_finish_s": float(np.median(t_succ)) if t_succ else None,
                "intervention_rate": float(np.mean([e["intervention_rate"] for e in eps])),
                "brake_override_rate": float(np.mean([e["brake_override_rate"] for e in eps])),
            }"""

new_analyze_row = """            mean_ns = float(np.mean([e.get("mean_n_surv", 0.0) for e in eps]))
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
            }"""

assert old_analyze_row in text
text = text.replace(old_analyze_row, new_analyze_row)

old_line = """            line = (f"{m:<22} n={n:<3} succ {row['success_pct']:5.1f} [{s_ci[0]:4.1f},{s_ci[1]:5.1f}] | "
                    f"unsafe {row['unsafe_pct']:5.1f} [{u_ci[0]:4.1f},{u_ci[1]:5.1f}] | "
                    f"stall {row['stalled_pct']:4.1f} tmo {row['timeout_pct']:4.1f} | "
                    f"v {row['mean_speed']:4.2f} T {ft} | intv {row['intervention_rate']:.2f} "
                    f"brk {row['brake_override_rate']:.2f}")"""

new_line = """            line = (f"{m:<22} n={n:<3} succ {row['success_pct']:5.1f} [{s_ci[0]:4.1f},{s_ci[1]:5.1f}] | "
                    f"unsafe {row['unsafe_pct']:5.1f} [{u_ci[0]:4.1f},{u_ci[1]:5.1f}] | "
                    f"stall {row['stalled_pct']:4.1f} tmo {row['timeout_pct']:4.1f} | "
                    f"v {row['mean_speed']:4.2f} T {ft} | intv {row['intervention_rate']:.2f} "
                    f"brk {row['brake_override_rate']:.2f} | "
                    f"n_surv {mean_ns:4.1f} lt5 {frac_lt5*100:4.1f}%")"""

assert old_line in text
text = text.replace(old_line, new_line)

with open('/home/raj/agile_nicbf/stress_grid.py', 'w') as f:
    f.write(text)

print('stress_grid.py patched successfully!')
