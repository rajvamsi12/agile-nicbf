import os
import sys
import time
import json
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from render_video import record_simulation, create_mp4_video

output_dir = os.path.join(current_dir, "recorded_videos_all_4")
os.makedirs(output_dir, exist_ok=True)

methods = ["mppi_alone", "analytical_hocbf", "neural_icbf", "agile_nicbf_coupled"]
seeds = [101, 138, 175, 212, 249]

all_results = {m: [] for m in methods}
video_records = []

print("=" * 100)
print("  20-Video Benchmark Pipeline (4 Methods x 5 Seeds) on GPU")
print("  Destination:", output_dir)
print("=" * 100 + "\n")

total_t0 = time.time()

for m_idx, method in enumerate(methods, 1):
    print(f"\n=== Method [{m_idx}/4]: {method} ===")
    for s_idx, seed in enumerate(seeds, 1):
        idx_global = (m_idx - 1) * len(seeds) + s_idx
        print(f"\n  [{idx_global:02d}/20] Simulating {method} (Seed {seed})...")
        t0 = time.time()
        
        frames, obstacles = record_simulation(
            method=method,
            seed=seed,
            max_steps=750,
            dt=0.02,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        sim_time = time.time() - t0
        
        last = frames[-1]
        final_x = float(last["x"])
        outcome = "success" if last.get("success", False) or final_x >= 50.0 else (
            "collision" if last.get("collision", False) else (
                "out_of_bounds" if last.get("out_of_bounds", False) else "timeout"
            )
        )
        steps = len(frames)
        avg_speed = float(np.mean([f["vx"] for f in frames]))
        interventions = sum(1 for f in frames if f.get("qp_active", False))
        int_rate = (interventions / max(1, steps)) * 100.0
        
        # Latency metrics if available
        qp_lats = [f.get("qp_latency_ms", 0.0) for f in frames if f.get("qp_latency_ms", 0.0) > 0]
        avg_qp_lat = float(np.mean(qp_lats)) if qp_lats else 0.0
        
        print(f"    Sim complete ({sim_time:.1f}s): {outcome.upper()} | Steps: {steps} ({steps*0.02:.2f}s) | Final X: {final_x:.2f}m | Avg Speed: {avg_speed:.2f} m/s")
        
        # Render MP4
        video_name = f"{method}_seed_{seed}_{outcome}.mp4"
        video_path = os.path.join(output_dir, video_name)
        print(f"    Rendering video to {video_name}...")
        
        t_vid = time.time()
        frames_sub = frames[::2] if len(frames) > 100 else frames
        create_mp4_video(frames_sub, obstacles, output_mp4=video_path, fps=25)
        vid_time = time.time() - t_vid
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"    Rendered in {vid_time:.1f}s ({size_mb:.2f} MB)")
        
        res_entry = {
            "method": method,
            "seed": seed,
            "outcome": outcome,
            "success": outcome == "success",
            "collision": outcome == "collision",
            "out_of_bounds": outcome == "out_of_bounds",
            "timeout": outcome == "timeout",
            "steps": steps,
            "duration_s": round(steps * 0.02, 2),
            "final_x": round(final_x, 2),
            "avg_speed_mps": round(avg_speed, 2),
            "interventions": interventions,
            "intervention_rate_pct": round(int_rate, 1),
            "video_filename": video_name,
            "video_path": video_path,
            "file_size_mb": round(size_mb, 2)
        }
        all_results[method].append(res_entry)
        video_records.append(res_entry)

# Wilson score interval helper
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - spread) * 100.0, min(1.0, center + spread) * 100.0

# Print formatted summary table
print("\n" + "=" * 130)
print(f"{'Method':<22} | {'Completed':<11} | {'Success (95% CI)':<20} | {'Collision (95% CI)':<20} | {'OOB':<6} | {'Timeout':<8} | {'Avg Speed':<10} | {'Videos Saved'}")
print("-" * 130)

summary_table = []
for m in methods:
    res_list = all_results[m]
    n = len(res_list)
    n_succ = sum(1 for r in res_list if r["success"])
    n_coll = sum(1 for r in res_list if r["collision"])
    n_oob = sum(1 for r in res_list if r["out_of_bounds"])
    n_time = sum(1 for r in res_list if r["timeout"])
    
    succ_low, succ_hi = wilson(n_succ, n)
    coll_low, coll_hi = wilson(n_coll, n)
    
    avg_spd = float(np.mean([r["avg_speed_mps"] for r in res_list]))
    succ_str = f"{n_succ/n*100:.1f}% [{succ_low:.1f}, {succ_hi:.1f}]"
    coll_str = f"{n_coll/n*100:.1f}% [{coll_low:.1f}, {coll_hi:.1f}]"
    oob_str = f"{n_oob/n*100:.1f}%"
    time_str = f"{n_time/n*100:.1f}%"
    comp_str = f"{n_succ}/{n} ({n_succ/n*100:.1f}%)"
    
    print(f"{m:<22} | {comp_str:<11} | {succ_str:<20} | {coll_str:<20} | {oob_str:<6} | {time_str:<8} | {avg_spd:.2f} m/s   | 5 MP4s ({output_dir})")
    
    summary_table.append({
        "method": m,
        "completed": comp_str,
        "success_pct": round(n_succ/n*100, 1),
        "success_ci": [round(succ_low, 1), round(succ_hi, 1)],
        "collision_pct": round(n_coll/n*100, 1),
        "collision_ci": [round(coll_low, 1), round(coll_hi, 1)],
        "oob_pct": round(n_oob/n*100, 1),
        "timeout_pct": round(n_time/n*100, 1),
        "avg_speed": round(avg_spd, 2)
    })

print("=" * 130)

# Save summary JSON
summary_payload = {
    "total_videos": len(video_records),
    "destination": output_dir,
    "elapsed_time_s": round(time.time() - total_t0, 1),
    "summary": summary_table,
    "runs": video_records
}

summary_json_path = os.path.join(output_dir, "summary_20_runs.json")
with open(summary_json_path, "w") as f:
    json.dump(summary_payload, f, indent=2)

print(f"\nAll 20 videos and summary saved successfully to {output_dir}")
