# Agile-NICBF: Neural Integral Control Barrier Functions for Agile Racing

The paper's Neural ICBF pipeline (learned dynamics F_θ, imitation policy π_θ, learned barriers h_θ, integral control law) adapted to a 6-D dynamic bicycle: F1TENTH parameters, tire-force saturation, steering slew and motor lag. The car races through a cluttered 50 m × 4 m track at **3.5–4.0 m/s**, and the nominal planner is GPU MPPI.

The code is in `prototype_neural_ode/dynamic/`. Earlier experiments at 6 m/s use different methods. They are kept for the record in [Earlier experiments at 6 m/s](#earlier-experiments-at-6-ms-not-the-papers-method).

---

## Results: Neural ICBF at 3.5 and 4.0 m/s

Setup: 30 seeded obstacle layouts, μ = 1.0, MPPI with K = 50 samples. Values are percentages. *Unsafe* means collision or out of bounds.

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

**Takeaways**
- Both Neural ICBF variants are far safer than the analytical HOCBF-QP: 0–3.3% unsafe vs. 20–33%.
- With MPPI as the nominal there were no unsafe episodes at either speed. Success matches MPPI alone at 3.5 m/s and is slightly lower at 4.0 m/s.
- The QP runs in 7–8 ms per step on the CPU.

**Caveats**
- With n = 30 the confidence intervals overlap. Neural ICBF does not significantly beat MPPI alone.
- The config was tuned on 16 of the 30 evaluation seeds (101–656). Seed 878, the one π_θ-nominal crash, is held out.
- π_θ ignores obstacles, as in the paper, so most failures are stalls (deadlocks in front of an obstacle).
- **Speed range.** At 3.5–4 m/s MPPI stays mostly below the grip limit (above 90% of available grip on only about 4–5% of steps). At higher speeds one-step barrier filters break down; see below.

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

Known issues in the original kinematic prototype (`prototype_neural_ode/*.py`):
- Its `safe_controller.py` uses an analytic kinematic `bicycle_f()` and never the learned F_θ.
- Its h is position-only.
- It loads `model/*_learned.pth`, which do not exist. A `model -> models` symlink was added; nothing was renamed.

---

## Earlier experiments at 6 m/s (not the paper's method)

These came before the Neural ICBF pipeline. They show why the method is restricted to sub-limit speeds. They also cover a different filter, the predictive filter, which works at 6 m/s but is **not** an ICBF. They are kept as a research record.

**Findings**
1. **One-step barrier QPs fail at the grip limit.** Once the front tire saturates, the steering gradient of the wall barrier changes sign (from about +20 to −5). The QP then steers *into* the wall and blocks MPPI's counter-steer, so the car spins and stalls. MPPI alone recovers on the same seeds. QP patches (`--wall-steer-guard`, `--obs-steer-guard`, `--slip-brake-limit`, all off by default) only trade stalls for crashes. The traces are in `diag/trace_*.json` and the reports in `logs/efg_*_report.log`.
2. **The original 6-D barrier network (`models/neural_icbf_6d.pth`) was used outside its training range.** It was trained on |r|, |v_y| ≤ 1, but MPPI exceeds that range on 64% of steps.
3. **A predictive filter works at 6 m/s.** Code: `predictive_filter.py`, `fast_rollout.py`. It checks MPPI's samples against a brake-to-stop backup maneuver and commits only to verified plans. It reaches 90–93% success with 0 unsafe outcomes at K = 1000. A learned terminal certificate (`train_terminal_barrier.py`) added neither safety nor speed.

Figures and a LaTeX table from the first benchmark version: `results/trajectory_comparison.png`, `results/friction_circle_compliance.png`, `results/benchmark_table.tex`.

All tables below use 30 seeds; values are percentages. Older 3–5-seed smoke grids (`grid_*`, `smoke*`) are kept in `results/` but are too small to report.

### A. Baseline shields at 6 and 8 m/s (K = 50)

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

## Repository layout

```
.
├── dynamic_bicycle_env.py, safe_controller.py, mppi_planner.py   # identical copies are kept in src/
├── predictive_filter.py, fast_rollout.py, train_terminal_barrier.py   # 6 m/s predictive filter (earlier work)
├── stress_grid.py               # benchmark driver: all methods, sharding, Wilson CIs, --analyze
├── run_*.sh                     # exact commands used for every reported run
├── src/                         # env, analytical/neural QP shields, MPPI, training, plotting, video
├── models/                      # earlier checkpoints (6-D/9-D barrier nets, terminal barrier)
├── prototype_neural_ode/        # kinematic prototype + dynamic/ (Neural ICBF pipeline, main result)
├── diag/                        # diagnostics, episode traces, predictive-filter configs
├── tests/                       # ad-hoc tests and debugging scripts
├── results/                     # per-episode JSONL, summaries, figures, demo video
└── logs/                        # console logs (logs/nicbf_overnight.log for the Neural ICBF run)
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

# Predictive filter (earlier 6 m/s work)
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
