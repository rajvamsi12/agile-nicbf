# Neural ICBF on the dynamic bicycle: overnight run report (2026-10-16)

Evaluation: 30 standard seeds at mu=1.0, K=50, v_target 3.5 and 4.0 m/s. Frozen config: {"alpha":5,"gamma":10,"a_rate_scale":30,"h_offset":0.2}.
Full table: results/nicbf_overnight_summary.json. Console log: logs/nicbf_overnight.log.

| v | Method | Success [95% CI] | Unsafe | Stall | Timeout | Latency |
|---|---|---|---|---|---|---|
| 3.5 | MPPI alone | 86.7 [70,95] | 0.0 | 13.3 | 0 | - |
| 3.5 | Analytical HOCBF | 60.0 [42,75] | 20.0 | 10.0 | 10.0 | - |
| 3.5 | NICBF, pi_theta nominal | 73.3 [56,86] | 3.3 (seed 878) | 23.3 | 0 | 7.8 ms |
| 3.5 | NICBF, MPPI nominal | 86.7 [70,95] | 0.0 | 13.3 | 0 | 7.2 ms |
| 4.0 | MPPI alone | 86.7 [70,95] | 6.7 | 6.7 | 0 | - |
| 4.0 | Analytical HOCBF | 33.3 [19,51] | 33.3 | 26.7 | 6.7 | - |
| 4.0 | NICBF, pi_theta nominal | 80.0 [63,91] | 3.3 (seed 878) | 16.7 | 0 | 7.9 ms |
| 4.0 | NICBF, MPPI nominal | 83.3 [66,93] | 0.0 | 16.7 | 0 | 7.3 ms |

Caveats:
- 16 of the 30 seeds (101-656) were used for tuning.
- Seed 878, the one NICBF crash, is held out. On that seed the learned obstacle barrier reached h=-1.09 while slack was active.
- At n=30 the confidence intervals overlap. NICBF clearly beats analytical HOCBF, but it does not beat MPPI alone.
- pi_theta does not see obstacles, so most NICBF failures are stalls (deadlocks in front of an obstacle).
- The obstacle barrier h_obs takes the obstacle radius as an extra 7th input.
- Prototype issues: the original prototype's filter used an analytic kinematic f(), not F_theta. Its model/*_learned.pth files do not exist; a symlink model -> models was added.
- Sync safe_controller.py, mppi_planner.py and dynamic_bicycle_env.py (root and src/) and stress_grid.py to the Mac.
