import os
import sys
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dynamic_bicycle_env import DynamicBicycleModel, DynamicBicycleParams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Neural Barrier Network Architecture (MLP)
# ---------------------------------------------------------------------------
class NeuralBarrierNet(nn.Module):
    """
    Neural ICBF State Barrier Network h_theta(x).
    Input: [dx_obs, dy_obs, psi_rel, vx, vy, r] (6D relative dynamic state)
    Output: scalar h(x) in [-1, 1] representing continuous dynamic safety margin.
    """
    def __init__(self, input_dim=6, hidden_dim=256):
        super(NeuralBarrierNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. Worst-Case Friction (mu in [0.4, 1.1]) & 3-Disk Footprint Rollout Labeler
# ---------------------------------------------------------------------------
def generate_robust_rollout_dataset(num_samples=150000, horizon_steps=30, dt=0.02,
                                    mu_grid=(0.40, 0.70, 1.05)):
    """
    Generates dynamic safety labels via multi-friction forward rollouts on GPU:
    1. Evaluates 49 time-varying evasive maneuvers.
    2. Checks 3-disk vehicle footprint (rear axle, CG, front axle) against obstacle.
    3. Evaluates escape across mu in mu_grid, taking the WORST-CASE clearance:
       d_worst(x0) = min_{mu} max_{u} min_{t} dist(x(t), obs).
    """
    print("=" * 80)
    print(f"[Stage 2] Generating {num_samples} Robust Rollout Samples on {device}")
    print(f"          Friction Grid mu: {mu_grid} (Worst-Case Robustness)")
    print(f"          Footprint: 3 Disks (r=0.21m) at rear, CG, and front axles")
    print("=" * 80)
    
    t_start = time.time()
    
    # Base model for parameters
    base_model = DynamicBicycleModel(device=device)
    p = base_model.p
    lr = p.lr
    lf = p.lf
    r_disk = 0.21
    r_obs = 0.30
    r_collision = r_obs + r_disk  # 0.51m
    
    # Sample relative states to obstacle at origin (0, 0)
    # dx: vehicle position x relative to obstacle (-5m to +5m)
    # dy: vehicle position y relative to obstacle (-2.5m to +2.5m)
    dx = (torch.rand(num_samples, device=device) * 10.0) - 5.0
    dy = (torch.rand(num_samples, device=device) * 5.0) - 2.5
    psi = (torch.rand(num_samples, device=device) * 2.0 * np.pi) - np.pi
    vx = torch.rand(num_samples, device=device) * 9.5 + 0.5   # 0.5 to 10.0 m/s
    vy = (torch.rand(num_samples, device=device) * 2.0) - 1.0 # lateral drift
    r = (torch.rand(num_samples, device=device) * 2.0) - 1.0   # yaw rate
    
    init_states = torch.stack([dx, dy, psi, vx, vy, r], dim=1)
    
    # Initial 3-disk collision check at t=0
    cos_p0 = torch.cos(psi)
    sin_p0 = torch.sin(psi)
    d_rear_0 = torch.sqrt((dx - lr * cos_p0)**2 + (dy - lr * sin_p0)**2)
    d_cg_0 = torch.sqrt(dx**2 + dy**2)
    d_front_0 = torch.sqrt((dx + lf * cos_p0)**2 + (dy + lf * sin_p0)**2)
    min_dist_0 = torch.minimum(d_cg_0, torch.minimum(d_rear_0, d_front_0))
    geometric_unsafe = min_dist_0 <= r_collision
    
    # 49 Time-varying action profiles: [Horizon, 2] per maneuver
    accels = [-p.a_max, -0.5 * p.a_max, 0.0, 0.5 * p.a_max, p.a_max]
    steers = [-p.delta_max, -0.5 * p.delta_max, 0.0, 0.5 * p.delta_max, p.delta_max]
    
    time_varying_maneuvers = []
    
    # 25 Constant full-effort profiles
    for a in accels:
        for s in steers:
            profile = torch.zeros((horizon_steps, 2), device=device)
            profile[:, 0] = a
            profile[:, 1] = s
            time_varying_maneuvers.append(profile)
            
    # 24 Two-Phase Time-Varying Maneuvers
    split_step = horizon_steps // 2
    for s in [-p.delta_max, -0.5 * p.delta_max, 0.5 * p.delta_max, p.delta_max]:
        for a in [-0.5 * p.a_max, 0.0, 0.5 * p.a_max]:
            # S-Curves: Steer s -> Counter-steer -s
            p1 = torch.zeros((horizon_steps, 2), device=device)
            p1[:split_step, 0] = a
            p1[:split_step, 1] = s
            p1[split_step:, 0] = a
            p1[split_step:, 1] = -s
            time_varying_maneuvers.append(p1)
            
            # Swerve-and-Straighten: Steer s -> Straighten 0
            p2 = torch.zeros((horizon_steps, 2), device=device)
            p2[:split_step, 0] = a
            p2[:split_step, 1] = s
            p2[split_step:, 0] = a
            p2[split_step:, 1] = 0.0
            time_varying_maneuvers.append(p2)
            
    num_maneuvers = len(time_varying_maneuvers)
    print(f"          Evaluating {num_maneuvers} action profiles across {len(mu_grid)} friction levels...")
    
    # Worst-case escape clearance across mu grid
    worst_case_max_clearance = torch.full((num_samples,), 99.0, device=device)
    worst_case_can_escape = torch.ones(num_samples, dtype=torch.bool, device=device)
    
    for mu_val in mu_grid:
        t_mu0 = time.time()
        # Instantiate model with this friction
        model_mu = DynamicBicycleModel(device=device)
        model_mu.p.mu = float(mu_val)
        
        # Track best evasive trajectory under this mu
        mu_max_min_dist = torch.zeros(num_samples, device=device)
        mu_can_escape = torch.zeros(num_samples, dtype=torch.bool, device=device)
        
        for profile in time_varying_maneuvers:
            curr_state = init_states.clone()
            trajectory_min_dist = torch.full((num_samples,), 99.0, device=device)
            trajectory_safe = torch.ones(num_samples, dtype=torch.bool, device=device)
            
            for step in range(horizon_steps):
                u_t = profile[step, :].unsqueeze(0).repeat(num_samples, 1)
                curr_state = model_mu.step_rk4_torch(curr_state, u_t, dt=dt)
                
                # 3-Disk footprint distance
                px_t = curr_state[:, 0]
                py_t = curr_state[:, 1]
                psi_t = curr_state[:, 2]
                cos_pt = torch.cos(psi_t)
                sin_pt = torch.sin(psi_t)
                
                d_rear = torch.sqrt((px_t - lr * cos_pt)**2 + (py_t - lr * sin_pt)**2)
                d_cg = torch.sqrt(px_t**2 + py_t**2)
                d_front = torch.sqrt((px_t + lf * cos_pt)**2 + (py_t + lf * sin_pt)**2)
                d_step = torch.minimum(d_cg, torch.minimum(d_rear, d_front))
                
                trajectory_min_dist = torch.minimum(trajectory_min_dist, d_step)
                trajectory_safe = trajectory_safe & (d_step > r_collision)
                
            mu_can_escape = mu_can_escape | trajectory_safe
            mu_max_min_dist = torch.maximum(mu_max_min_dist, trajectory_min_dist)
            
        elapsed_mu = time.time() - t_mu0
        escaped_pct = (mu_can_escape.sum().item() / num_samples) * 100.0
        print(f"          mu={mu_val:.2f} completed in {elapsed_mu:.1f}s | Escapable: {escaped_pct:.1f}%")
        
        # Take pointwise minimum over mu (worst-case escape)
        worst_case_max_clearance = torch.minimum(worst_case_max_clearance, mu_max_min_dist)
        worst_case_can_escape = worst_case_can_escape & mu_can_escape

    # Smooth continuous barrier target:
    # h = tanh((worst_clearance - r_collision) / scale)
    # Inside collision / inescapable crash zone: h < 0
    # Safe escape zone: h > 0, scaling smoothly with clearance margin
    barrier_margin = (worst_case_max_clearance - r_collision) / 1.2
    labels = torch.where(geometric_unsafe, torch.tensor(-1.0, device=device),
                         torch.where(worst_case_can_escape, torch.tanh(barrier_margin), torch.tensor(-0.8, device=device)))
    labels = labels.unsqueeze(1)
    
    elapsed_total = time.time() - t_start
    safe_count = (labels > 0.0).sum().item()
    unsafe_count = (labels <= 0.0).sum().item()
    print(f"[Stage 2] Rollouts Complete in {elapsed_total:.1f}s")
    print(f"          Robust Safe States: {safe_count} ({safe_count/num_samples*100:.1f}%)")
    print(f"          Unsafe/Escape-Zone States: {unsafe_count} ({unsafe_count/num_samples*100:.1f}%)")
    
    return init_states.detach(), labels.detach()


# ---------------------------------------------------------------------------
# 3. Model Training Loop
# ---------------------------------------------------------------------------
def train_robust_neural_icbf(num_samples=150000, epochs=30, batch_size=1024, lr=1e-3,
                             save_path="/home/raj/agile_nicbf/models/neural_icbf_6d.pth"):
    """
    Trains Neural Barrier Network with worst-case mu robustness on GPU.
    """
    print("\n" + "=" * 80)
    print(f"[Stage 3] Training Robust Neural ICBF on Device: {device}")
    print(f"          Target Destination: {save_path}")
    print("=" * 80)
    
    states, labels = generate_robust_rollout_dataset(num_samples=num_samples, mu_grid=(0.40, 0.70, 1.05))
    
    n_train = int(0.85 * len(states))
    train_dataset = TensorDataset(states[:n_train], labels[:n_train])
    val_dataset = TensorDataset(states[n_train:], labels[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = NeuralBarrierNet(input_dim=6, hidden_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Back up existing checkpoint
    if os.path.exists(save_path):
        backup_path = save_path.replace(".pth", "_nominal_backup.pth")
        shutil.copy2(save_path, backup_path)
        print(f"[Stage 3] Backed up previous checkpoint to {backup_path}")
        
    t_train0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
            
        train_loss /= len(train_dataset)
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                val_loss += criterion(preds, batch_y).item() * len(batch_x)
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            mark = "(* Best Saved)"
        else:
            mark = ""
            
        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e} {mark}")
            
    total_time = time.time() - t_train0
    print("=" * 80)
    print(f"[Stage 3] Training Complete in {total_time:.1f}s | Best Val MSE: {best_val_loss:.5f}")
    print(f"          Robust Barrier Model Saved: {save_path}")
    print("=" * 80)

if __name__ == "__main__":
    train_robust_neural_icbf()
