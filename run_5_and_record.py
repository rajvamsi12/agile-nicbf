import os
import sys
import time
import json
import numpy as np
import torch
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from render_video import record_simulation, create_mp4_video

output_dir = os.path.join(current_dir, "recorded_videos")
os.makedirs(output_dir, exist_ok=True)

seeds = [101, 138, 175, 212, 249]
results = []

print("=" * 66)
print("  Agile-NICBF 5-Episode Benchmark Video Recording")
print("  Method: agile_nicbf_coupled (Coupled Barrier MPPI + Shield QP)")
print("  Seeds:", seeds)
print("  Destination Folder:", output_dir)
print("=" * 66 + "\n")

for i, seed in enumerate(seeds, 1):
    print(f"--- [Episode {i}/5] Simulating Seed {seed} ---")
    t0 = time.time()
    frames, obstacles = record_simulation(
        method="agile_nicbf_coupled",
        seed=seed,
        max_steps=750,
        dt=0.02,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    sim_time = time.time() - t0
    
    last = frames[-1]
    final_x = last["x"]
    outcome = "success" if last.get("success", False) or final_x >= 50.0 else (
        "collision" if last.get("collision", False) else (
            "out_of_bounds" if last.get("out_of_bounds", False) else "timeout"
        )
    )
    steps = len(frames)
    avg_speed = float(np.mean([f["vx"] for f in frames]))
    interventions = sum(1 for f in frames if f["qp_active"])
    int_rate = interventions / max(1, steps) * 100.0
    
    print(f"  Result: {outcome.upper()} | Steps: {steps} ({steps*0.02:.2f}s) | Final X: {final_x:.2f}m | Avg Speed: {avg_speed:.2f} m/s | QP: {int_rate:.1f}%")
    
    video_filename = f"run_{i}_seed_{seed}_{outcome}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    
    t_vid = time.time()
    frames_sub = frames[::2] if len(frames) > 100 else frames
    create_mp4_video(frames_sub, obstacles, output_mp4=video_path, fps=25)
    render_time = time.time() - t_vid
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"  Video rendered: {video_filename} ({size_mb:.2f} MB in {render_time:.1f}s)\n")
    
    results.append({
        "run": i,
        "seed": seed,
        "method": "agile_nicbf_coupled",
        "outcome": outcome,
        "steps": steps,
        "duration_s": round(steps * 0.02, 2),
        "final_x": round(float(final_x), 2),
        "avg_speed_mps": round(avg_speed, 2),
        "interventions": interventions,
        "intervention_rate_pct": round(int_rate, 1),
        "video_filename": video_filename,
        "video_path": video_path,
        "file_size_mb": round(size_mb, 2)
    })

summary_path = os.path.join(output_dir, "summary_5_runs.json")
with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)

print("=" * 66)
print("  5-Episode Recording Complete!")
print(f"  Summary saved: {summary_path}")
print("=" * 66)
