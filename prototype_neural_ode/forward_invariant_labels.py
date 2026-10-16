import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def generate_labels(points, pi_theta, centers, radii,
                    T_label=20, dt=0.1, n_headings=8, u_max=4.0,
                    dynamics_fn=None, in_any_circle_fn=None):
    """
    Vectorized rollout labeling.
    points: torch.Tensor, shape (N, 2)
    pi_theta: NNController model
    """
    device = next(pi_theta.parameters()).device
    points = points.to(device)
    N = points.shape[0]
    
    # 1. Build batch tensor of shape (N * n_headings, 3)
    points_tiled = points.repeat_interleave(n_headings, dim=0)
    
    # Sample n_headings evenly-spaced headings in [0, 2pi)
    headings = torch.arange(0, n_headings, device=device, dtype=points.dtype) * (2.0 * np.pi / n_headings)
    headings_tiled = headings.repeat(N)
    
    state_batch = torch.cat([points_tiled, headings_tiled.unsqueeze(1)], dim=1)
    B = N * n_headings
    
    # Initialize ever_unsafe mask
    ever_unsafe = torch.zeros(B, dtype=torch.bool, device=device)
    
    # Check if initial states are already inside any obstacle
    for center, radius in zip(centers, radii):
        dist = (state_batch[:, :2] - center.to(device)).norm(dim=1)
        ever_unsafe = ever_unsafe | (dist < radius)
        
    # 2. Rollout loop
    for step in tqdm(range(T_label), desc="Forward-invariance rollout", unit="step"):
        # Query controller
        u = pi_theta(state_batch)
        if u_max is not None:
            u_norm = torch.linalg.norm(u, dim=1, keepdim=True)
            u = torch.where(u_norm > u_max, u * (u_max / u_norm), u)
            
        # Step dynamics
        state_batch = dynamics_fn(state_batch, u, dt)
        
        # Check obstacles
        step_unsafe = torch.zeros(B, dtype=torch.bool, device=device)
        for center, radius in zip(centers, radii):
            dist = (state_batch[:, :2] - center.to(device)).norm(dim=1)
            step_unsafe = step_unsafe | (dist < radius)
            
        ever_unsafe = ever_unsafe | step_unsafe
        
    # 3. Reshape and compute final labels
    ever_unsafe = ever_unsafe.view(N, n_headings)
    new_labels = torch.ones(N, dtype=torch.float32, device=device)
    new_labels[ever_unsafe.any(dim=1)] = -1.0
    
    # Compute old purely geometric labels for comparison
    old_labels = torch.ones(N, dtype=torch.float32, device=device)
    old_unsafe = torch.zeros(N, dtype=torch.bool, device=device)
    for center, radius in zip(centers, radii):
        dist = (points - center.to(device)).norm(dim=1)
        old_unsafe = old_unsafe | (dist < radius)
    old_labels[old_unsafe] = -1.0
    
    # Compute flips
    flipped_to_unsafe = torch.where((old_labels == 1.0) & (new_labels == -1.0))[0]
    flipped_to_safe = torch.where((old_labels == -1.0) & (new_labels == 1.0))[0]
    
    tqdm.write(f"Labeling complete: {len(flipped_to_unsafe)}/{N} points relabeled safe->unsafe ({100*len(flipped_to_unsafe)/N:.2f}%)")
    
    return new_labels, old_labels, flipped_to_unsafe, flipped_to_safe

def plot_label_comparison(points, old_labels, new_labels, centers, radii,
                            save_path="outputs/label_comparison.png"):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    points_np = points.cpu().numpy()
    old_labels_np = old_labels.cpu().numpy()
    new_labels_np = new_labels.cpu().numpy()
    
    # Categorize points
    safe_safe = (old_labels_np == 1.0) & (new_labels_np == 1.0)
    unsafe_unsafe = (old_labels_np == -1.0) & (new_labels_np == -1.0)
    safe_unsafe = (old_labels_np == 1.0) & (new_labels_np == -1.0)
    unsafe_safe = (old_labels_np == -1.0) & (new_labels_np == 1.0)
    
    plt.figure(figsize=(8, 8))
    
    # Scatter plot with specific colors
    plt.scatter(points_np[safe_safe, 0], points_np[safe_safe, 1], c='green', s=10, label='Safe (Both)', alpha=0.6)
    plt.scatter(points_np[unsafe_unsafe, 0], points_np[unsafe_unsafe, 1], c='red', s=10, label='Unsafe (Both)', alpha=0.6)
    plt.scatter(points_np[safe_unsafe, 0], points_np[safe_unsafe, 1], c='orange', s=15, label='Safe -> Unsafe (Forward-Invariant Catch)', alpha=0.8)
    plt.scatter(points_np[unsafe_safe, 0], points_np[unsafe_safe, 1], c='blue', s=15, label='Unsafe -> Safe (Rare)', alpha=0.8)
    
    # Draw obstacles
    ax = plt.gca()
    for center, radius in zip(centers, radii):
        circle = plt.Circle((center[0].item(), center[1].item()), radius, color='black', fill=False, linewidth=2, linestyle='--')
        ax.add_patch(circle)
        
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('x')
    plt.ylabel('y')
    
    blue_count = np.sum(unsafe_safe)
    orange_count = np.sum(safe_unsafe)
    title = f"Label Comparison (Orange/Flipped: {orange_count})"
    if blue_count > 0:
        title += f" [Warning: {blue_count} Unsafe->Safe]"
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Safe (Both): {np.sum(safe_safe)}")
    print(f"Unsafe (Both): {np.sum(unsafe_unsafe)}")
    print(f"Safe -> Unsafe: {orange_count}")
    print(f"Unsafe -> Safe: {blue_count}")
