"""Held-out 200-seed summary: table with Wilson CIs + paired exact McNemar tests vs MPPI alone
(results/nicbf_heldout200.jsonl -> results/nicbf_heldout200_summary.json)."""
import sys, os, json, collections
from math import comb
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."));  sys.path[:0] = [root, os.path.join(root, "src")]
import numpy as np
from stress_grid import load, wilson

R = load([os.path.join(root, "results", "nicbf_heldout200.jsonl")])
methods = ["mppi_alone", "analytical_hocbf", "nicbf", "nicbf_mppi"]
labels = {"mppi_alone": "MPPI alone", "analytical_hocbf": "Analytical HOCBF-QP",
          "nicbf": "Neural ICBF, pi_theta nominal", "nicbf_mppi": "Neural ICBF, MPPI nominal"}
UNSAFE = ("collision", "out_of_bounds")


def mcnemar(b, c):
    """Exact two-sided McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


summary = {}
pct = lambda p, ci: f"{p*100:5.1f} [{ci[0]:4.1f},{ci[1]:4.1f}]"
print(f"{'v':>4} {'method':30s} {'n':>4} {'success':>18} {'unsafe':>18} {'stall':>6} {'tmo':>5} {'lat ms':>7}")
for v in (3.5, 4.0):
    by = {m: {r["seed"]: r for r in R if r["method"] == m and r["v_target"] == v} for m in methods}
    common = sorted(set.intersection(*[set(d) for d in by.values() if d])) if any(by.values()) else []
    for m in methods:
        X = list(by[m].values())
        if not X:
            continue
        n = len(X); c = collections.Counter(r["outcome"] for r in X)
        uns = sum(c[o] for o in UNSAFE)
        ws, wu = wilson(c["success"], n), wilson(uns, n)
        lat = [r["mean_latency_ms"] for r in X if "mean_latency_ms" in r]
        row = {"n": n, "success": c["success"] / n, "success_ci_pct": ws, "unsafe": uns / n, "unsafe_ci_pct": wu,
               "collision": c["collision"] / n, "out_of_bounds": c["out_of_bounds"] / n,
               "stalled": c["stalled"] / n, "timeout": c["timeout"] / n,
               "mean_speed": float(np.mean([r["mean_speed"] for r in X])),
               "mean_latency_ms": float(np.mean(lat)) if lat else None}
        if m != "mppi_alone" and common:
            a, b = by["mppi_alone"], by[m]
            for key, test in (("unsafe", lambda r: r["outcome"] in UNSAFE), ("success", lambda r: r["outcome"] == "success")):
                only_ref = sum(test(a[s]) and not test(b[s]) for s in common)
                only_m = sum(test(b[s]) and not test(a[s]) for s in common)
                row[f"paired_{key}_vs_mppi"] = {"n_common": len(common), "mppi_only": only_ref, "method_only": only_m,
                                                "mcnemar_p": mcnemar(only_ref, only_m)}
        summary[f"v{v}_{m}"] = row
        print(f"{v:4.1f} {labels[m]:30s} {n:4d} {pct(row['success'], ws):>18} {pct(row['unsafe'], wu):>18} "
              f"{row['stalled']*100:6.1f} {row['timeout']*100:5.1f} {(row['mean_latency_ms'] or 0):7.1f}")
    print()
print("Paired vs MPPI alone (same seeds; exact McNemar):")
for k, row in summary.items():
    for key in ("unsafe", "success"):
        p = row.get(f"paired_{key}_vs_mppi")
        if p:
            print(f"  {k:28s} {key:8s}: only MPPI {p['mppi_only']:3d} | only method {p['method_only']:3d} | p = {p['mcnemar_p']:.4f}")
json.dump(summary, open(os.path.join(root, "results", "nicbf_heldout200_summary.json"), "w"), indent=2)
print("\nwrote results/nicbf_heldout200_summary.json")
