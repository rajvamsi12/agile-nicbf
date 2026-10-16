import os
import json

def _latency_ms(stats):
    return stats.get("avg_latency_ms", stats.get("avg_qp_latency_ms", 0.0))


def generate_latex_table(json_path="results/benchmark_comparison.json", output_tex="results/benchmark_table.tex"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    
    # LaTeX Table Template (L4DC / IEEE Transactions format)
    latex_code = []
    latex_code.append(r"\begin{table}[t]")
    latex_code.append(r"\centering")
    latex_code.append(r"\caption{Four-method benchmark and ablation on the cluttered autonomous racing corridor ($50\,\mathrm{m}$, 10 obstacles, $v_{\mathrm{target}}=6.0\,\mathrm{m/s}$).}")
    latex_code.append(r"\label{tab:benchmark_results}")
    latex_code.append(r"\begin{tabular}{lcccccccc}")
    latex_code.append(r"\toprule")
    latex_code.append(r"\textbf{Method} & \textbf{Success (\%)} $\uparrow$ & \textbf{Collision (\%)} $\downarrow$ & \textbf{OOB (\%)} $\downarrow$ & \textbf{Timeout (\%)} $\downarrow$ & \textbf{Avg Speed (m/s)} $\uparrow$ & \textbf{Time (s)} $\downarrow$ & \textbf{QP Lat (ms)} $\downarrow$ & \textbf{Total Lat (ms)} $\downarrow$ \\")
    latex_code.append(r"\midrule")

    method_metadata = {
        "mppi_alone": "MPPI Alone (No Shield)",
        "analytical_hocbf": "Analytical HOCBF-QP",
        "neural_icbf": "Decoupled Neural ICBF-QP",
        "agile_nicbf_coupled": r"\textbf{Agile-NICBF (Coupled, Ours)}"
    }

    order = ["mppi_alone", "analytical_hocbf", "neural_icbf", "agile_nicbf_coupled"]

    for key in order:
        if key not in summary:
            continue
        stats = summary[key]
        name = method_metadata.get(key, key)
        succ_val = stats.get('success_rate_pct', 0.0)
        coll_val = stats.get('collision_rate_pct', 0.0)
        oob_val = stats.get('oob_rate_pct', 0.0)
        timeout_val = max(0.0, 100.0 - succ_val - coll_val - oob_val)
        
        # Include Wilson confidence intervals if present
        if "success_ci_95" in stats:
            ci = stats["success_ci_95"]
            succ = f"{succ_val:.1f}\\% \\tiny{{[{ci[0]:.1f}, {ci[1]:.1f}]}}"
        else:
            succ = f"{succ_val:.1f}\\%"

        if "collision_ci_95" in stats:
            ci = stats["collision_ci_95"]
            coll = f"{coll_val:.1f}\\% \\tiny{{[{ci[0]:.1f}, {ci[1]:.1f}]}}"
        else:
            coll = f"{coll_val:.1f}\\%"

        oob = f"{oob_val:.1f}\\%"
        tout = f"{timeout_val:.1f}\\%"
        speed = f"{stats.get('avg_speed_mps', 0.0):.2f}"
        t_comp = f"{stats.get('avg_completion_time_s', 0.0):.2f}" if stats.get('avg_completion_time_s', 0.0) > 0 else "--"
        
        qp_lat_val = stats.get('avg_qp_latency_ms', 0.0)
        qp_lat = f"{qp_lat_val:.2f}" if qp_lat_val > 0 else "--"
        
        tot_lat_val = stats.get('avg_total_latency_ms', stats.get('avg_latency_ms', 0.0))
        tot_lat = f"{tot_lat_val:.2f}" if tot_lat_val > 0 else "--"

        latex_code.append(f"{name} & {succ} & {coll} & {oob} & {tout} & {speed} & {t_comp} & {qp_lat} & {tot_lat} \\\\")

    latex_code.append(r"\bottomrule")
    latex_code.append(r"\end{tabular}")
    latex_code.append(r"\end{table}")

    table_str = "\n".join(latex_code)
    os.makedirs(os.path.dirname(output_tex), exist_ok=True)
    with open(output_tex, "w") as f:
        f.write(table_str)

    print(f"\n[Stage 6] Generated 3-method LaTeX table at {output_tex}:\n")
    print(table_str)
    return table_str

if __name__ == "__main__":
    generate_latex_table()
