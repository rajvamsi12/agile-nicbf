"""Final summary for the overnight Neural ICBF run: table + paired comparison + JSON (results/nicbf_overnight_summary.json)."""
import sys, os, json, collections
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."));  sys.path[:0] = [root, os.path.join(root, "src")]
import numpy as np
from stress_grid import load, wilson

R = load([os.path.join(root, "results", "nicbf_eval_baselines.jsonl"), os.path.join(root, "results", "nicbf_eval_pipeline.jsonl")])
methods = ["mppi_alone", "analytical_hocbf", "nicbf", "nicbf_mppi"]
labels = {"mppi_alone": "MPPI alone", "analytical_hocbf": "Analytical HOCBF (MPPI nominal)",
          "nicbf": "Neural ICBF, pi_theta nominal (paper)", "nicbf_mppi": "Neural ICBF, MPPI nominal"}
summary = {}
print(f"{'v':>4} {'method':38s} {'n':>3} {'success':>16} {'collision':>9} {'OOB':>5} {'UNSAFE':>16} {'stall':>6} {'tmo':>5} {'speed':>6} {'lat ms':>7}")
for v in (3.5, 4.0):
    for m in methods:
        X = [r for r in R if r["method"] == m and r["v_target"] == v]
        if not X:
            continue
        n = len(X); c = collections.Counter(r["outcome"] for r in X)
        uns = c["collision"] + c["out_of_bounds"]
        ws, wu = wilson(c["success"], n), wilson(uns, n)
        lat = [r["mean_latency_ms"] for r in X if "mean_latency_ms" in r]
        row = {"n": n, "success": c["success"] / n, "success_ci_pct": ws, "collision": c["collision"] / n,
               "out_of_bounds": c["out_of_bounds"] / n, "unsafe": uns / n, "unsafe_ci_pct": wu, "stalled": c["stalled"] / n,
               "timeout": c["timeout"] / n, "mean_speed": float(np.mean([r["mean_speed"] for r in X])),
               "mean_latency_ms": float(np.mean(lat)) if lat else None,
               "failures": sorted((r["seed"], r["outcome"]) for r in X if r["outcome"] != "success")}
        summary[f"v{v}_{m}"] = row
        f = lambda p, ci: f"{p*100:5.1f} [{ci[0]:4.1f},{ci[1]:4.1f}]" if ci else f"{p*100:5.1f}"
        print(f"{v:4.1f} {labels[m]:38s} {n:3d} {f(row['success'], ws):>16} {row['collision']*100:9.1f} {row['out_of_bounds']*100:5.1f} "
              f"{f(row['unsafe'], wu):>16} {row['stalled']*100:6.1f} {row['timeout']*100:5.1f} {row['mean_speed']:6.2f} "
              f"{(row['mean_latency_ms'] or 0):7.1f}")
    print()
# paired: NICBF vs MPPI alone on the same seeds
for v in (3.5, 4.0):
    base = {r["seed"]: r["outcome"] for r in R if r["method"] == "mppi_alone" and r["v_target"] == v}
    for m in ("nicbf", "nicbf_mppi"):
        X = {r["seed"]: r["outcome"] for r in R if r["method"] == m and r["v_target"] == v}
        if not X:
            continue
        pairs = collections.Counter((X[s], base.get(s)) for s in X)
        summary[f"v{v}_{m}_paired_vs_mppi"] = {f"{a}|mppi:{b}": k for (a, b), k in pairs.items()}
        print(f"v={v} {m} vs mppi_alone (same seeds):", dict(sorted(((f'{a} | mppi {b}'), k) for (a, b), k in pairs.items())))
json.dump(summary, open(os.path.join(root, "results", "nicbf_overnight_summary.json"), "w"), indent=2, default=str)
print("\nwrote results/nicbf_overnight_summary.json")
