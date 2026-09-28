import os
import json
import numpy as np

def generate_latex_table(json_path="results/benchmark_comparison.json", output_tex="results/benchmark_table.tex"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    
    # LaTeX Table Template (L4DC / IEEE Transactions format)
    latex_code = []
    latex_code.append(r"\begin{table*}[t]")
    latex_code.append(r"\centering")
    latex_code.append(r"\caption{State-of-the-Art Comparative Benchmark on Cluttered Autonomous Racing Corridor ($50\,\mathrm{m}$, 20 Obstacles, $v_{\mathrm{target}}=6.0\,\mathrm{m/s}$).}")
    latex_code.append(r"\label{tab:sota_benchmark_results}")
    latex_code.append(r"\begin{tabular}{llcccccc}")
    latex_code.append(r"\toprule")
    latex_code.append(r"\textbf{Paradigm} & \textbf{Method / Reference} & \textbf{Success (\%)} $\uparrow$ & \textbf{Collision (\%)} $\downarrow$ & \textbf{OOB (\%)} $\downarrow$ & \textbf{Avg Speed (m/s)} $\uparrow$ & \textbf{Time (s)} $\downarrow$ & \textbf{Latency (ms)} $\downarrow$ \\")
    latex_code.append(r"\midrule")

    method_metadata = {
        "mppi_alone": ("Sampling MPC", "MPPI Alone (Williams et al., 2017)"),
        "analytical_hocbf": ("Classical CBF", "Analytical HOCBF-QP (Xiao \& Belta, 2021)"),
        "mpcc_liniger": ("Contouring MPC", "Nonlinear MPCC (Liniger et al., 2015)"),
        "sfc_minco": ("Robotics Splines", "SFC / MINCO (Wang \& Gao, 2022)"),
        "drl_policy": ("Deep RL", "End-to-End DRL (Wurman et al., 2022)"),
        "neural_icbf": ("Learned Shield", r"\textbf{Agile-NICBF (Ours)}")
    }

    order = ["mppi_alone", "analytical_hocbf", "mpcc_liniger", "sfc_minco", "drl_policy", "neural_icbf"]

    for key in order:
        if key not in summary:
            continue
        stats = summary[key]
        paradigm, name = method_metadata.get(key, ("Other", key))
        succ = f"{stats.get('success_rate_pct', 0.0):.1f}\\%"
        coll = f"{stats.get('collision_rate_pct', 0.0):.1f}\\%"
        oob = f"{stats.get('oob_rate_pct', 0.0):.1f}\\%"
        speed = f"{stats.get('avg_speed_mps', 0.0):.2f}"
        t_comp = f"{stats.get('avg_completion_time_s', 0.0):.2f}" if stats.get('avg_completion_time_s', 0.0) > 0 else "--"
        lat = f"{stats.get('avg_latency_ms', 0.0):.2f}" if stats.get('avg_latency_ms', 0.0) > 0 else "--"

        latex_code.append(f"{paradigm} & {name} & {succ} & {coll} & {oob} & {speed} & {t_comp} & {lat} \\\\")

    latex_code.append(r"\bottomrule")
    latex_code.append(r"\end{tabular}")
    latex_code.append(r"\end{table*}")

    table_str = "\n".join(latex_code)
    os.makedirs(os.path.dirname(output_tex), exist_ok=True)
    with open(output_tex, "w") as f:
        f.write(table_str)

    print(f"\n[Stage 6] Generated 6-Way SOTA LaTeX Table at {output_tex}:\n")
    print(table_str)
    return table_str

if __name__ == "__main__":
    generate_latex_table()
