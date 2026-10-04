import os
import json
import numpy as np

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
    latex_code.append(r"\caption{Three-method benchmark on the cluttered autonomous racing corridor ($50\,\mathrm{m}$, 20 obstacles, $v_{\mathrm{target}}=6.0\,\mathrm{m/s}$).}")
    latex_code.append(r"\label{tab:benchmark_results}")
    latex_code.append(r"\begin{tabular}{lcccccc}")
    latex_code.append(r"\toprule")
    latex_code.append(r"\textbf{Method} & \textbf{Success (\%)} $\uparrow$ & \textbf{Collision (\%)} $\downarrow$ & \textbf{OOB (\%)} $\downarrow$ & \textbf{Avg Speed (m/s)} $\uparrow$ & \textbf{Time (s)} $\downarrow$ & \textbf{QP Latency (ms)} $\downarrow$ \\")
    latex_code.append(r"\midrule")

    method_metadata = {
        "mppi_alone": "MPPI Alone (No Shield)",
        "analytical_hocbf": "Analytical HOCBF-QP",
        "neural_icbf": r"\textbf{Agile-NICBF (Ours)}"
    }

    order = ["mppi_alone", "analytical_hocbf", "neural_icbf"]

    for key in order:
        if key not in summary:
            continue
        stats = summary[key]
        name = method_metadata.get(key, key)
        succ = f"{stats.get('success_rate_pct', 0.0):.1f}\\%"
        coll = f"{stats.get('collision_rate_pct', 0.0):.1f}\\%"
        oob = f"{stats.get('oob_rate_pct', 0.0):.1f}\\%"
        speed = f"{stats.get('avg_speed_mps', 0.0):.2f}"
        t_comp = f"{stats.get('avg_completion_time_s', 0.0):.2f}" if stats.get('avg_completion_time_s', 0.0) > 0 else "--"
        latency = _latency_ms(stats)
        lat = f"{latency:.2f}" if latency > 0 else "--"

        latex_code.append(f"{name} & {succ} & {coll} & {oob} & {speed} & {t_comp} & {lat} \\\\")

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
