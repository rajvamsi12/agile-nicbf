# Agile-NICBF: Neural Integral Control Barrier Functions for Agile Racing

Safety filters for a 6-D dynamic bicycle (F1TENTH parameters, tire-force saturation, steering slew and motor lag) racing through a cluttered 50 m × 4 m track. The nominal planner is GPU MPPI. This repository holds the full project history:

| Phase | Where | What |
|---|---|---|
| 1 | `prototype_neural_ode/*.py` | Original kinematic Neural-ODE / imitation / Neural CBF prototype |
| 2 | `src/`, root `*.py`, `models/` | Dynamic-bicycle environment, MPPI, analytical HOCBF and neural-barrier QP shields, benchmark suite |
| 3 | `stress_grid.py`, `run_efg_*`, `diag/` | Stress-grid benchmark and failure diagnostics of the HOCBF-QP shield |
| 4 | `predictive_filter.py`, `fast_rollout.py`, `train_terminal_barrier.py` | Predictive (gatekeeper) safety filter using MPPI samples and a brake-to-stop backup |
| 5 | `prototype_neural_ode/dynamic/` | Paper's Neural ICBF pipeline (Algorithm 1) adapted to the dynamic bicycle |

The development log is in `PROJECT_LOG_AND_METHODOLOGY.md`.

---

## Headline results

All results use the same seeded obstacle layouts (30 seeds, μ = 1.0), reported as percentages. *Unsafe* means collision or out of bounds. Every episode is in `results/*.jsonl`.

### Neural ICBF pipeline (Phase 5), v_target 3.5 and 4.0 m/s, K = 50

| v | Method | Success [95% Wilson CI] | Unsafe | Stall | Timeout | Filter latency |
|---|---|---|---|---|---|---|
| 3.5 | MPPI alone | 86.7 [70, 95] | 0.0 | 13.3 | 0 | – |
| 3.5 | Analytical HOCBF-QP | 60.0 [42, 75] | 20.0 | 10.0 | 10.0 | – |
| 3.5 | Neural ICBF, π_θ nominal (paper) | 73.3 [56, 86] | 3.3 | 23.3 | 0 | 7.8 ms |
| 3.5 | Neural ICBF, MPPI nominal | 86.7 [70, 95] | 0.0 | 13.3 | 0 | 7.2 ms |
| 4.0 | MPPI alone | 86.7 [70, 95] | 6.7 | 6.7 | 0 | – |
| 4.0 | Analytical HOCBF-QP | 33.3 [19, 51] | 33.3 | 26.7 | 6.7 | – |
| 4.0 | Neural ICBF, π_θ nominal (paper) | 80.0 [63, 91] | 3.3 | 16.7 | 0 | 7.9 ms |
| 4.0 | Neural ICBF, MPPI nominal | 83.3 [66, 93] | 0.0 | 16.7 | 0 | 7.3 ms |

Sources: `results/nicbf_eval_baselines.jsonl`, `results/nicbf_eval_pipeline.jsonl`, `results/nicbf_overnight_summary.json`, `results/nicbf_overnight_REPORT.md`.

### Predictive filter (Phase 4), v_target 6.0 m/s

| Method | K | Success | Unsafe | Stall |
|---|---|---|---|---|
| MPPI alone | 50 | 53.3 | 33.3 | 13.3 |
| MPPI alone | 1000 | 93.3 | 3.3 | 3.3 |
| Analytical HOCBF-QP | 50 | 0.0 | 23.3 | 66.7 (+10 timeout) |
| Predictive filter v9 | 50 | 90.0 | 3.3 | 6.7 |
| Predictive filter v9 | 1000 | 93.3 | 0.0 | 6.7 |
| Predictive filter v10x (exact, numpy, 15.6 ms/step) | 1000 | 90.0 | 0.0 | 10.0 |

### Caveats

- With n = 30 the confidence intervals overlap. Both filters clearly beat the analytical HOCBF-QP, but neither significantly beats MPPI alone.
- The Neural ICBF configuration was tuned on 16 of the 30 evaluation seeds (101–656). Seed 878, the one π_θ-nominal crash, is held out.
- π_θ ignores obstacles, as in the paper, so most Neural ICBF failures are stalls (deadlocks in front of an obstacle).

---

## Full results

All tables use 30 seeds and the same seeded layouts, reported as percentages. Unsafe means collision or out of bounds. The source file in `results/` is listed under each table. Older 3–5-seed smoke grids (`grid_*`, `smoke*`) are kept in `results/` but are too small to report.

### A. Baseline shields at 6 and 8 m/s (Phases 2–3, K = 50)

| μ | v | MPPI alone | MPPI + oracle μ | Analytical HOCBF-QP | Neural ICBF-QP (6-D net) | Agile-NICBF coupled |
|---|---|---|---|---|---|---|
| 1.0 | 6 | **53.3** / 33.3 / 13.3 | 56.7 / 33.4 / 10.0 | 0.0 / 23.3 / 66.7 | 3.3 / 26.6 / 66.7 | 3.3 / 26.7 / 63.3 |
| 1.0 | 8 | **33.3** / 53.3 / 13.3 | 33.3 / 56.7 / 10.0 | 0.0 / 43.3 / 56.7 | – | 0.0 / 43.4 / 56.7 |
| 0.7 | 6 | 0.0 / 90.0 / 10.0 | **10.0** / 46.7 / 43.3 | 0.0 / 23.3 / 76.7 | – | 0.0 / 33.4 / 63.3 |
| 0.7 | 8 | 0.0 / 90.0 / 10.0 | **10.0** / 66.6 / 23.3 | 0.0 / 50.0 / 50.0 | – | 0.0 / 60.0 / 40.0 |

Each cell is success / unsafe / stall; the remainder is timeout. Sources: `ablate_clamp_on`, `part_e_trigger_logging`. With brake clamping off (`ablate_clamp_off`) the shield rows are identical.

### B. Effect of planner sample count K (μ = 1.0, v = 6)

| Method | K | Success | Unsafe | Stall | Timeout |
|---|---|---|---|---|---|
| MPPI alone | 50 | 53.3 | 33.3 | 13.3 | 0 |
| MPPI alone | 1000 | **93.3** | 3.3 | 3.3 | 0 |
| Analytical HOCBF-QP | 50 | 0.0 | 23.3 | 66.7 | 10.0 |
| Analytical HOCBF-QP | 200 | 6.7 | 43.3 | 30.0 | 20.0 |
| Analytical HOCBF-QP | 1000 | 43.3 | 20.0 | 6.7 | 30.0 |
| Neural ICBF-QP (6-D net) | 1000 | 13.3 | 50.0 | 6.7 | 30.0 |
| Agile-NICBF coupled | 1000 | 10.0 | 43.3 | 10.0 | 36.7 |

Sources: `ceiling_mppi_k1000`, `ablation_*_k*`.

### C. HOCBF-QP shield ablations and patches (μ = 1.0, v = 6, K = 50)

| Variant | Success | Unsafe | Stall | Timeout |
|---|---|---|---|---|
| Default analytical HOCBF-QP | 0.0 | 23.3 | 66.7 | 10.0 |
| Wall rows removed | 23.3 | 66.7 | 10.0 | 0 |
| Wall α = 50 | 20.0 | 70.0 | 10.0 | 0 |
| Wall α = 80 | 16.7 | 76.7 | 6.7 | 0 |
| Safety margin r = 0.21 / 0.26 / 0.30 m | 0.0 / 0.0 / 0.0 | 30.0 / 23.3 / 20.0 | 63.3 / 63.3 / 70.0 | 6.7 / 13.3 / 10.0 |
| `--wall-steer-guard zero_steer` | 6.7 | 33.4 | 60.0 | 0 |
| `--wall-steer-guard drop_row` | 40.0 | 60.0 | 0 | 0 |
| Wall + obstacle steer guards (options 1+2) | **56.7** | 43.3 | 0 | 0 |
| Same guards, Neural ICBF-QP | 36.7 | 60.0 | 3.3 | 0 |
| Same guards, Agile-NICBF coupled | 46.7 | 43.3 | 10.0 | 0 |

Removing or patching the wall rows trades stalls for crashes. None of the patches makes the one-step QP both safe and live. Sources: `ablation_hocbf_*`, `sweep_rmargin_*`, `wallguard_*`, `opt12_guard`.

### D. Predictive filter versions (μ = 1.0, v = 6)

| Version | Main change | K = 50: S / U / St | K = 1000: S / U / St |
|---|---|---|---|
| v1 | multi-step predictive filter, no backup | 53.3 / 23.4 / 23.3 | – |
| v2 | + MPPI samples as candidates, robust progress selection | 80.0 / 6.7 / 13.3 | 80.0 / 10.0 / 6.7 |
| v3 | + brake-to-stop backup | 66.7 / 16.7 / 16.7 | – |
| v4 | + carry-over of the committed plan | 70.0 / **0.0** / 30.0 | 83.3 / **0.0** / 16.7 |
| v6 | + heading limit 1.4 rad, yaw-damping backup | **96.7** / 3.3 / 0.0 | 90.0 / **0.0** / 10.0 |
| v7 | short 15-step backup + neural terminal certificate | 86.7 / 3.3 / 10.0 | 83.3 / 6.6 / 10.0 |
| v8 | neural screen, exact full-stop verify of top 8 | 86.7 / 6.7 / 6.7 | 86.7 / 3.3 / 10.0 |
| v9 | + 0.1 m commitment margin, carried plan re-verified | 90.0 / 3.3 / 6.7 | **93.3 / 0.0** / 6.7 |
| v10 | numpy backend, neural screen + exact verify | 76.7 / 10.0 / 13.3 | **93.3 / 0.0** / 6.7 |
| v10x | exact screen, no network (15.6 ms/step) | 86.7 / 10.0 / 3.3 | 90.0 / **0.0** / 10.0 |
| v10x, assumes μ = 0.95 | robustness margin on grip | 93.3 / 0.0 / 6.7 | 86.7 / 3.3 / 10.0 |

S = success, U = unsafe, St = stall. Sources: `predictive_filter`, `psf_v*`. Configs are in `diag/cfg_v10*.json`. Differences of 1–3 episodes out of 30 are within noise.

---

## Main findings

1. **Why the HOCBF-QP shield fails at the grip limit.** Once the front tire saturates, the steering gradient of the one-step wall barrier changes sign (from about +20 to −5). The QP then steers *into* the wall and blocks MPPI's counter-steer, so the car spins and stalls. MPPI alone recovers on the same seeds. QP patches (`--wall-steer-guard`, `--obs-steer-guard`, `--slip-brake-limit`, all off by default) did not fix this. The traces are in `diag/trace_*.json` and the reports in `logs/efg_*_report.log`.
2. **The original 6-D barrier network was used outside its training range.** It was trained on |r|, |v_y| ≤ 1, but MPPI exceeds that range on 64% of steps.
3. **A predictive filter works at 6 m/s.** It checks MPPI's samples against a brake-to-stop backup maneuver and commits only to verified plans. It reaches 90–93% success with 0 unsafe outcomes at K = 1000. A learned terminal certificate added neither safety nor speed.
4. **The paper's Neural ICBF works on the dynamic bicycle at sub-limit speeds (3.5–4 m/s).** It uses learned dynamics F_θ, viability-labelled learned barriers, and an integral control law with ∂F/∂u computed by autograd. It is far safer than the analytical HOCBF and about as good as MPPI alone. With MPPI as the nominal it had no unsafe episodes.

---

## Neural ICBF pipeline (`prototype_neural_ode/dynamic/`)

| Stage | Script | Output (`dynamic/models/`) | Validation |
|---|---|---|---|
| 1. Learned dynamics F_θ (Neural ODE, RK4, multi-step loss) | `stage1_dynamics.py` | `F_theta.pth` | 1 s rollout position error: 8.5 cm median, 14 cm p95. ∂F/∂u: cosine 0.99998, 8.4% median relative error |
| 2. Imitation policy π_θ (GPU MPPI expert, empty track, v_target as input) | `stage2_imitation.py collect --shard i/n`, then `train` | `pi_theta.pth`, `il_data_*.npz` | Mean absolute error: a_x 0.235 m/s², δ 0.062 rad |
| 3. Learned barriers h_obs(dx, dy, ψ, v_x, v_y, r, r_obs) and h_wall | `stage3_barrier.py obs` and `stage3_barrier.py wall` | `h_obs.pth`, `h_wall.pth` | Accuracy 99.7% / 99.4%. No unsafe validation states scored as safe at the 0.2 margin |
| 4. Neural ICBF QP (integral law u̇ = φ + v*, grip constraint ≤ 0.8 μg, actuator range and rate limits) | `nicbf_controller.py` | – | 6–10 ms/step on CPU |
| 5. Evaluation | `../../run_nicbf_final_eval.sh`, `summarize_overnight.py` | `results/nicbf_*` | See the table above |

- Barrier labels are viability labels: a state is labelled safe if at least one of 22 backup maneuvers, simulated through F_θ with the actuator model, keeps the car clear for 1.5 s. The bottom wall reuses h_wall by mirror symmetry.
- The frozen config is in `nicbf_final_config.json`: `{"alpha": 5, "gamma": 10, "a_rate_scale": 30, "h_offset": 0.2}`.
- The model directory can be overridden with the `NICBF_MODELS` environment variable.

Known issues in the Phase 1 prototype:
- Its `safe_controller.py` uses an analytic kinematic `bicycle_f()` and never the learned F_θ.
- Its h is position-only.
- It loads `model/*_learned.pth`, which do not exist. A `model -> models` symlink was added; nothing was renamed.

---

## Paper draft (L4DC 2027) vs. this repository

| Paper item | In this repo? | Where / note |
|---|---|---|
| Vehicle F_θ, Neural ODE (Sec. 4.1, Fig. 2) | Code yes, figure no | `prototype_neural_ode/system_neural_ode.py`, `learn_all.py`. No script produces Fig. 2 |
| Imitation π_θ from NMPC via do-mpc (Sec. 4.2) | Code yes | `prototype_neural_ode/imitation_learner.py`. `do_mpc` is not in `requirements.txt` |
| Forward-invariant labels, N_θ = 8, T = 20 (Sec. 5.2) | Yes | `prototype_neural_ode/forward_invariant_labels.py`, `learn_all.py` |
| h_θ and ICBF QP (Sec. 4.3–4.4, Algorithm 1) | Yes | `prototype_neural_ode/neural_cbf.py`, `safe_controller.py`, `main.py`. The filter uses the analytic kinematic f₁, not F_θ |
| Fig. 3 (h and h_e projections) | No | No plotting script |
| Fig. 4 (geometric vs. forward-invariant labels) | Partly | `learn_all.py` writes `outputs/h_theta_heatmap_forward_invariant.png`. The image itself is not in the repo |
| Table 4 (geometric vs. forward-invariant violations) | Code yes, results no | `prototype_neural_ode/main_compare.py`. It needs `model/ICBF_model_learned.pth` and `..._v2_forward_invariant.pth`, which are missing, and its output CSV is not saved |
| Table 2 (vehicle: NMPC / ORCA-MAPF / CADRL) | **No** | No ORCA-MAPF, CADRL or timing/cost benchmark code |
| Table 3 (quadrotor-pendulum) | **No** | No quadrotor-pendulum code or models |
| Table 5 (SABLAS, double integrator) | **No** | No SABLAS or double-integrator code |
| Neural ICBF on the dynamic bicycle (not in the draft yet) | Yes | `prototype_neural_ode/dynamic/`. See the results above |

Prototype checkpoints that are present: `prototype_neural_ode/models/{system_model*.pth, IL_model*.pth, ICBF_model.pth}`. Loss curves are in `losses_*.csv`.

---

## Repository layout

```
.
├── dynamic_bicycle_env.py, safe_controller.py, mppi_planner.py   # identical copies are kept in src/
├── predictive_filter.py, fast_rollout.py, train_terminal_barrier.py
├── stress_grid.py               # benchmark driver: all methods, sharding, Wilson CIs, --analyze
├── run_*.sh                     # exact commands used for every reported run
├── src/                         # Phase 2 package (env, shields, MPPI, training, plotting, video)
├── models/                      # Phase 2–4 checkpoints (6-D/9-D barrier nets, terminal barrier)
├── prototype_neural_ode/        # Phase 1 prototype + dynamic/ (Phase 5 Neural ICBF pipeline)
├── diag/                        # diagnostics, episode traces, predictive-filter configs
├── tests/                       # ad-hoc tests and debugging scripts
├── results/                     # per-episode JSONL, summaries, figures, demo video
└── logs/                        # console logs (logs/nicbf_overnight.log for Phase 5)
```

**Sync rule:** `safe_controller.py`, `dynamic_bicycle_env.py` and `mppi_planner.py` must stay byte-identical between the repo root and `src/`.

---

## Quick start

```bash
git clone https://github.com/rajvamsi12/agile-nicbf.git && cd agile-nicbf
conda create -n agile_nicbf python=3.10 -y && conda activate agile_nicbf
pip install -r requirements.txt          # tested with torch 2.13 (CUDA 13.0), numpy 2.2, qpsolvers 4.13
```

Benchmarks (`stress_grid.py`):

```bash
# Baselines
python stress_grid.py --mu 1.0 --speeds 3.5 4.0 --Ks 50 --episodes 30 \
    --methods mppi_alone analytical_hocbf --out results/my_baselines.jsonl

# Neural ICBF (pretrained models in prototype_neural_ode/dynamic/models)
python stress_grid.py --mu 1.0 --speeds 3.5 4.0 --Ks 50 --episodes 30 --methods nicbf nicbf_mppi \
    --nicbf "$(cat prototype_neural_ode/dynamic/nicbf_final_config.json)" --out results/my_nicbf.jsonl

# Predictive filter
python stress_grid.py --mu 1.0 --speeds 6 --Ks 1000 --episodes 30 --methods predictive_filter \
    --psf "$(cat diag/cfg_v10x.json)" --out results/my_psf.jsonl

# Summaries
python stress_grid.py --analyze results/my_nicbf.jsonl
python prototype_neural_ode/dynamic/summarize_overnight.py
```

- Add `--shard i/n` to split the seeds across processes.
- The `run_*.sh` scripts are the exact launchers used. They hard-code the workstation paths (`/home/raj/...`), so edit `cd` and `PY` before running them elsewhere.

Retrain the Neural ICBF pipeline:

```bash
cd prototype_neural_ode/dynamic
python stage1_dynamics.py
for i in 0 1 2 3; do python stage2_imitation.py collect --shard $i/4 & done; wait
python stage2_imitation.py train
python stage3_barrier.py obs && python stage3_barrier.py wall
python dev_eval.py nicbf 4.0 101,138 "$(cat nicbf_final_config.json)"   # quick check: method, speed, seeds, config
```

Not included in the repo (excluded by `.gitignore`):
- `f1tenth_gym/`: clone it from https://github.com/f1tenth/f1tenth_gym
- recorded video folders
- the 10 MB latency-replay tensor `diag/lat_rec_138_k1000.pt`

---

## Citation

```bibtex
@inproceedings{vamsi2027nicbf,
  title  = {Neural Integral Control Barrier Functions for Agile Collision Avoidance},
  author = {Vamsi, Dasari Raj and Zinage, Vrushabh and Bakolas, Efstathios and Chandra, Rohan},
  booktitle = {Learning for Dynamics and Control Conference (L4DC)},
  year   = {2027}
}
```

## License

MIT
