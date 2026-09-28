# Agile-NICBF: Microsecond Neural Integral Control Barrier Functions for Agile Autonomous Racing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 13.2](https://img.shields.io/badge/CUDA-13.2-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for **Agile-NICBF**: A microsecond Neural Integral Control Barrier Function (NICBF) safety shield integrated with GPU-accelerated Model Predictive Path Integral (MPPI) control for high-speed ($>6.0\,\mathrm{m/s}$, $22\,\mathrm{km/h}$) autonomous racing collision avoidance under dynamic tire slip and friction limits.

This repository contains the full project evolution:
1. **Phase 1 (`prototype_neural_ode/`):** The foundational Neural ODE and Imitation Learning research prototype for kinematic obstacle avoidance.
2. **Phase 2 (`src/` & `models/`):** The complete 6D Dynamic Bicycle Agile-NICBF framework with Pacejka tire dynamics, GPU MPPI planning, certified microsecond QP shield, and 6-way SOTA comparative benchmark suite.

---

## Key Features

- **6D Dynamic Single-Track Vehicle Dynamics:** Exact Pacejka $\tanh$ tire cornering stiffness model with load-transfer-balanced Kamm's friction circle limits ($F_{x}^2 + F_{y}^2 \le (\mu F_z)^2$).
- **Microsecond Active-Set QP Safety Shield:** Real-time online quadratic programming filter executing in $<3.0\,\mathrm{ms}$ with exact autograd Lie derivatives and emergency braking fail-safes.
- **GPU-Accelerated Reachability Training:** Neural Barrier Network trained on 250,000 GPU forward rollouts on NVIDIA T1000 across 49 time-varying action profiles (Val Loss: 0.0068, Accuracy: 97.18%).
- **Exhaustive 6-Way SOTA Benchmark Suite:** Evaluated against published baselines:
  1. *MPPI Alone* (Williams et al., IEEE T-RO 2017)
  2. *Analytical HOCBF-QP* (Xiao & Belta, IEEE TAC 2021)
  3. *Nonlinear MPCC* (Liniger et al., IEEE TCST 2015)
  4. *Safe Flight Corridors / MINCO Splines* (Wang & Gao, IEEE T-RO 2022)
  5. *Deep RL Racing Policy* (Wurman et al., Nature 2022)
  6. *Agile-NICBF (Ours)*

---

## Repository Structure

```
.
├── src/                               # Phase 2: 6D Dynamic Agile-NICBF Production Code
│   ├── dynamic_bicycle_env.py         # 6D F1TENTH vehicle physics & 50m cluttered racing track
│   ├── safe_controller.py             # Microsecond Agile-NICBF Active-Set QP safety shield
│   ├── mppi_planner.py                # GPU-accelerated MPPI parallel trajectory planner
│   ├── mpcc_controller.py             # Liniger MPCC contouring control baseline
│   ├── sfc_planner.py                 # Safe Flight Corridor / MINCO spline baseline
│   ├── drl_racing_policy.py           # Deep RL racing policy baseline
│   ├── train_neural_icbf.py           # 250k GPU dataset generation & neural barrier training
│   ├── benchmark_compare.py           # Master 6-way comparative benchmark suite
│   ├── render_video.py                # High-resolution MP4 video animation renderer
│   ├── plot_trajectories.py           # 2D racing line & telemetry plotting
│   └── generate_latex_table.py        # L4DC publication LaTeX table generator
│
├── models/                            # Phase 2 Trained Model Checkpoints
│   └── neural_icbf_6d.pth             # GPU-trained 6D Neural Barrier Network (Val Loss: 0.0068)
│
├── results/                           # Publication Figures, Videos & Tables
│   ├── agile_nicbf_demo.mp4           # 50m full-course racing video animation (30 FPS)
│   ├── trajectory_comparison.png      # 3-panel racing line & clearance figure
│   ├── friction_circle_compliance.png # Kamm friction circle compliance plot
│   ├── benchmark_comparison.json      # Full raw multi-seed benchmark metrics
│   └── benchmark_table.tex            # Publication-ready LaTeX table
│
├── prototype_neural_ode/              # Phase 1: Foundational Research Prototype
│   ├── system_neural_ode.py           # Neural ODE system identification
│   ├── imitation_learner.py           # Imitation learning baseline policy
│   ├── forward_invariant_labels.py    # Initial 2D forward-invariant label generator
│   ├── neural_cbf.py                  # Early 2D kinematic barrier network
│   ├── safe_controller.py             # Initial prototype safety controller
│   ├── main.py                        # Prototype closed-loop runner
│   ├── main_compare.py                # Prototype comparison script
│   ├── learn_all.py                   # Master pipeline trainer
│   └── models/                        # Prototype model checkpoints (.pth)
│
└── PROJECT_LOG_AND_METHODOLOGY.md     # Full development & methodology log (Steps 1–18)
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/rajvamsi12/agile-nicbf.git
cd agile-nicbf

# Create environment
conda create -n agile_nicbf python=3.10 -y
conda activate agile_nicbf

# Install dependencies
pip install torch numpy scipy qpsolvers quadprog osqp matplotlib
```

### Running the Agile-NICBF Safety Shield

```bash
python src/safe_controller.py
```

### Running the 6-Way SOTA Benchmark Suite

```bash
# Run 10-episode benchmark across all 6 baselines on GPU
python src/benchmark_compare.py --episodes 10 --device cuda
```

### Rendering the Full 50m Racing Video

```bash
python src/render_video.py
```

---

## Citation

```bibtex
@inproceedings{chandra2026agilenicbf,
  title={Microsecond Neural Integral Control Barrier Functions for Agile Autonomous Racing Collision Avoidance},
  author={Chandra, Rohan and Dasari, Raj Vamsi and DeepMind Team},
  booktitle={Learning for Dynamics and Control Conference (L4DC)},
  year={2026}
}
```
