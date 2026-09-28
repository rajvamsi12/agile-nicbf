import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure model unpickling classes are defined in __main__
import __main__
from safe_controller import SimpleNN, BicycleODEFunc, NNController
__main__.SimpleNN = SimpleNN
__main__.BicycleODEFunc = BicycleODEFunc
__main__.NNController = NNController

from safe_controller import compute_u_safe, u_N, h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', type=float, default=None, help='clip_v_norm value')
    args = parser.parse_args()
    
    dt = 0.1
    num_iterations = 100
    alpha = 0.1
    alpha_phi = 0.5
    gamma_e = 0.1
    gamma_u = 0.1
    u_max = 4.0
    clip_v_norm = args.clip

    state = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    
    # Initialize u from pi_theta at t=0 only
    u_N.eval()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    control = u_N(state_tensor).detach().numpy().flatten()

    states = []
    controls = []
    h_values = []
    hu_values = []
    he_values = []
    v_star_norms = []
    phi_norms = []
    
    diverged = False
    divergence_step = -1

    for step in range(num_iterations):
        states.append(state.copy())
        controls.append(control.copy())

        u_dot, phi, v_star, h_val, hu_val, he_val = compute_u_safe(
            state, control,
            alpha=alpha, alpha_phi=alpha_phi,
            gamma_e=gamma_e, gamma_u=gamma_u, u_max=u_max
        )

        v_norm = np.linalg.norm(v_star)
        v_star_norms.append(v_norm)
        phi_norms.append(np.linalg.norm(phi))
        h_values.append(h_val)
        hu_values.append(hu_val)
        he_values.append(he_val)

        # Optional clipping (report if active, do not hide it)
        if clip_v_norm is not None and v_norm > clip_v_norm:
            print(f"Step {step}: v* clipped from {v_norm:.3f} to {clip_v_norm:.3f}")
            v_star = v_star * (clip_v_norm / v_norm)
            u_dot = phi + v_star

        # INTEGRAL UPDATE: u carries forward
        control = control + u_dot * dt
        # Project onto input constraint set
        u_norm = np.linalg.norm(control)
        if u_norm > u_max:
            control = control * (u_max / u_norm)

        # Step true dynamics
        x, y, theta = state
        v_ctrl, delta = control[0], control[1]
        state = np.array([
            x + v_ctrl * np.cos(theta) * dt,
            y + v_ctrl * np.sin(theta) * dt,
            theta + delta * dt
        ], dtype=np.float32)
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi

        if step % 10 == 0:
            print(f"Step {step:3d} | h(x)={h_val:.4f} | hu(u)={hu_val:.4f} | he={he_val:.4f} | |v*|={v_norm:.4f}")

        # Check for divergence (any |state| > 100)
        if np.any(np.abs(state) > 100.0):
            diverged = True
            divergence_step = step + 1
            print(f"\nDIVERGENCE DETECTED at step {divergence_step}!")
            print(f"Diverging state: {state}")
            print(f"Diverging control: {control}")
            if len(states) >= 2:
                print(f"State 2 steps before divergence (step {divergence_step - 2}): {states[-2]}")
                print(f"Control 2 steps before divergence (step {divergence_step - 2}): {controls[-2]}")
            break

    # Add final state and control to lists if we did not diverge
    if not diverged:
        # One last print/record to match loop steps if desired, but we only ran num_iterations steps
        pass

    # Safety statistics
    steps_run = len(states)
    h_violations = sum(1 for v in h_values if v < 0)
    hu_violations = sum(1 for v in hu_values if v < 0)
    he_violations = sum(1 for v in he_values if v < 0)
    print(f"\n=== SAFETY SUMMARY ({steps_run} steps run) ===")
    print(f"h(x) violations (state):  {h_violations}/{steps_run} = {100*h_violations/steps_run:.1f}%")
    print(f"hu(u) violations (input): {hu_violations}/{steps_run} = {100*hu_violations/steps_run:.1f}%")
    print(f"he violations:            {he_violations}/{steps_run} = {100*he_violations/steps_run:.1f}%")
    
    if len(h_values) > 0:
        print(f"h(x) bounds:  min={min(h_values):.4f}, max={max(h_values):.4f}")
        print(f"hu(u) bounds: min={min(hu_values):.4f}, max={max(hu_values):.4f}")
        print(f"he bounds:    min={min(he_values):.4f}, max={max(he_values):.4f}")

    # Plotting
    os.makedirs('outputs', exist_ok=True)
    steps_range = range(steps_run)

    # Plot 1: x, y, theta vs step (3 subplots)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(steps_range, [s[0] for s in states], color='#1f77b4', linewidth=2)
    axs[0].set_ylabel('x', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].set_title('States over steps', fontsize=14)

    axs[1].plot(steps_range, [s[1] for s in states], color='#2ca02c', linewidth=2)
    axs[1].set_ylabel('y', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7)

    axs[2].plot(steps_range, [s[2] for s in states], color='#d62728', linewidth=2)
    axs[2].set_ylabel('theta', fontsize=12)
    axs[2].set_xlabel('Steps', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('outputs/plot_1_states.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: x-y trajectory with obstacle circle drawn at center (0.5, -0.5) radius 0.2 in orange
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([s[0] for s in states], [s[1] for s in states], color='#9467bd', linewidth=2.5, label='Trajectory')
    obstacle = plt.Circle((0.5, -0.5), 0.2, color='orange', fill=True, alpha=0.6, label='Obstacle')
    ax.add_patch(obstacle)
    
    # Also draw start position
    ax.plot(states[0][0], states[0][1], 'go', markersize=10, label='Start')
    if not diverged:
        ax.plot(state[0], state[1], 'ro', markersize=10, label='End')

    ax.set_title('Trajectory in x-y plane', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    plt.savefig('outputs/plot_2_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: h(x), hu(u), he vs step with a horizontal red dashed line at 0 labeled "Safety Boundary"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps_range, h_values, color='#1f77b4', linewidth=2, label='h(x) (state barrier)')
    ax.plot(steps_range, hu_values, color='#ff7f0e', linewidth=2, label='hu(u) (input barrier)')
    ax.plot(steps_range, he_values, color='#2ca02c', linewidth=2, label='he(x, u) (state barrier derivative filter)')
    ax.axhline(0.0, color='red', linestyle='--', linewidth=1.5, label='Safety Boundary')
    ax.set_title('Barrier Functions over steps', fontsize=14)
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Barrier Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    plt.savefig('outputs/plot_3_barriers.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: control inputs v and delta vs step
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(steps_range, [c[0] for c in controls], color='#1f77b4', linewidth=2)
    axs[0].set_ylabel('v (velocity)', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].set_title('Controls over steps', fontsize=14)

    axs[1].plot(steps_range, [c[1] for c in controls], color='#ff7f0e', linewidth=2)
    axs[1].set_ylabel('delta (steering angle)', fontsize=12)
    axs[1].set_xlabel('Steps', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('outputs/plot_4_controls.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nSaved 4 plots to the outputs/ directory.")

if __name__ == '__main__':
    main()