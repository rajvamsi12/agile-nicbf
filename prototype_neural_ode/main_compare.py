import os
import sys
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# Ensure model unpickling classes are defined in __main__
import __main__
from safe_controller import SimpleNN, BicycleODEFunc, NNController
__main__.SimpleNN = SimpleNN
__main__.BicycleODEFunc = BicycleODEFunc
__main__.NNController = NNController

import safe_controller
from safe_controller import compute_u_safe, u_N

def run_simulation(model_path, desc):
    # Load the specific h model and swap it in safe_controller module
    h_model = torch.load(model_path, weights_only=False)
    h_model.eval()
    safe_controller.h = h_model
    
    dt = 0.1
    num_iterations = 1000
    alpha = 0.1
    alpha_phi = 0.5
    gamma_e = 0.1
    gamma_u = 0.1
    u_max = 4.0

    state = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    
    u_N.eval()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    control = u_N(state_tensor).detach().numpy().flatten()

    states = []
    controls = []
    h_values = []
    hu_values = []
    he_values = []
    
    diverged = False
    divergence_step = -1

    for step in tqdm(range(num_iterations), desc=f"Running simulation ({desc})"):
        states.append(state.copy())
        controls.append(control.copy())

        u_dot, phi, v_star, h_val, hu_val, he_val = compute_u_safe(
            state, control,
            alpha=alpha, alpha_phi=alpha_phi,
            gamma_e=gamma_e, gamma_u=gamma_u, u_max=u_max
        )

        h_values.append(h_val)
        hu_values.append(hu_val)
        he_values.append(he_val)

        # INTEGRAL UPDATE: u carries forward
        control = control + u_dot * dt
        # Project onto input constraint set
        u_norm = np.linalg.norm(control)
        if u_norm > u_max:
            control = control * (u_max / u_norm)

        # Step dynamics
        x, y, theta = state
        v_ctrl, delta = control[0], control[1]
        state = np.array([
            x + v_ctrl * np.cos(theta) * dt,
            y + v_ctrl * np.sin(theta) * dt,
            theta + delta * dt
        ], dtype=np.float32)
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi

        if np.any(np.abs(state) > 100.0):
            diverged = True
            divergence_step = step + 1
            break

    steps_run = len(states)
    h_violations = sum(1 for v in h_values if v < 0)
    hu_violations = sum(1 for v in hu_values if v < 0)
    he_violations = sum(1 for v in he_values if v < 0)
    
    h_violations_pct = 100 * h_violations / steps_run
    hu_violations_pct = 100 * hu_violations / steps_run
    he_violations_pct = 100 * he_violations / steps_run
    
    return h_violations_pct, hu_violations_pct, he_violations_pct, diverged, divergence_step

def main():
    old_model_path = "model/ICBF_model_learned.pth"
    new_model_path = "model/ICBF_model_learned_v2_forward_invariant.pth"
    
    if not os.path.exists(old_model_path):
        print(f"Error: {old_model_path} not found.")
        sys.exit(1)
        
    if not os.path.exists(new_model_path):
        print(f"Error: {new_model_path} not found. Please run learn_all.py first to train it.")
        sys.exit(1)

    print("\nEvaluating OLD (geometric labeling) model...")
    old_h_pct, old_hu_pct, old_he_pct, old_div, old_div_step = run_simulation(old_model_path, "OLD Model")
    
    print("\nEvaluating NEW (forward-invariant labeling) model...")
    new_h_pct, new_hu_pct, new_he_pct, new_div, new_div_step = run_simulation(new_model_path, "NEW Model")
    
    print("\n" + "="*80)
    print(f"{'SAFETY METRICS SIDE-BY-SIDE':^80}")
    print("="*80)
    print(f"{'Violation Metric':<30} | {'OLD (geometric labeling)':<23} | {'NEW (forward-invariant)':<23}")
    print("-"*80)
    
    old_h_str = f"{old_h_pct:.2f}%" if not old_div else f"Diverged @ step {old_div_step}"
    new_h_str = f"{new_h_pct:.2f}%" if not new_div else f"Diverged @ step {new_div_step}"
    
    print(f"{'h(x) violations (state)':<30} | {old_h_str:<23} | {new_h_str:<23}")
    print(f"{'hu(u) violations (input)':<30} | {old_hu_pct:.2f}%:<23 | {new_hu_pct:.2f}%:<23")
    print(f"{'he violations':<30} | {old_he_pct:.2f}%:<23 | {new_he_pct:.2f}%:<23")
    print("="*80)
    
    # Save comparison to outputs/safety_comparison.csv
    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame([
        {
            "labeling_method": "OLD (geometric labeling)",
            "h_violations_pct": old_h_pct if not old_div else f"Diverged @ {old_div_step}",
            "hu_violations_pct": old_hu_pct,
            "he_violations_pct": old_he_pct
        },
        {
            "labeling_method": "NEW (forward-invariant labeling)",
            "h_violations_pct": new_h_pct if not new_div else f"Diverged @ {new_div_step}",
            "hu_violations_pct": new_hu_pct,
            "he_violations_pct": new_he_pct
        }
    ])
    df.to_csv("outputs/safety_comparison.csv", index=False)
    print("Saved side-by-side comparison to outputs/safety_comparison.csv\n")

if __name__ == "__main__":
    main()
