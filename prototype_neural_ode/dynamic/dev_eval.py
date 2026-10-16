"""Quick development evaluation: a few seeds of one NICBF config, prints outcomes + forensics."""
import sys, os, json, collections
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."));  sys.path[:0] = [root, os.path.join(root, "src")]
import numpy as np, torch
torch.set_num_threads(1)
import stress_grid as sg
method, v, seeds, kw = sys.argv[1], float(sys.argv[2]), [int(s) for s in sys.argv[3].split(",")], json.loads(sys.argv[4])
res = []
for s in seeds:
    r = sg.run_episode(method, s, 1.0, v, 50, "cuda" if torch.cuda.is_available() else "cpu", shield_kwargs={"nicbf": kw})
    res.append(r)
    print(f"  seed {s:5d}: {r['outcome']:<13} t={r['time_s']:5.2f}s v={r['mean_speed']:.2f} slack={r.get('frac_slack_used',0):.2f} "
          f"minh_obs={r.get('min_min_h_obs',0):+.2f} minh_wall={r.get('min_min_h_wall',0):+.2f} grip_viol={r.get('frac_grip_viol',0):.2f} "
          f"lat={r.get('mean_latency_ms',0):.1f}ms", flush=True)
c = collections.Counter(r["outcome"] for r in res)
print(f"{method} v={v} {kw}: " + " ".join(f"{k}={n}" for k, n in c.most_common()), flush=True)
