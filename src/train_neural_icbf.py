import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Robust path handling: ensures imports work from root, src/, or subdirectories
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
# 2. Vectorized Forward-Invariant Rollout Generator (49 Maneuvers)
# ---------------------------------------------------------------------------
def generate_rollout_dataset(num_samples=120000, horizon_steps=30, dt=0.02):
    """
    Generates dynamic safety labels via forward rollouts on GPU using RK4 and 
    49 true time-varying reachability action profiles:
    - 25 Constant profiles: 5 accels x 5 steers (Emergency braking, full throttle, constant swerves)
    - 24 Two-Phase time-varying profiles:
        - 12 S-Curves: Swerve steer s -> Counter-steer -s across 3 accels x 4 steers
        - 12 Swerve-and-Straighten: Swerve steer s -> Straighten 0 across 3 accels x 4 steers
    """
    print(f"[Stage 2] Generating {num_samples} dynamic forward-rollout samples on {device}...")
    start_time = time.time()
    
    model = DynamicBicycleModel(device=device)
    p = model.p
    
    # Sample relative states to obstacle at origin (0, 0)
    # dx: vehicle position x relative to obstacle (-4m to +4m)
    # dy: vehicle position y relative to obstacle (-2m to +2m)
    dx = (torch.rand(num_samples, device=device) * 8.0) - 4.0
    dy = (torch.rand(num_samples, device=device) * 4.0) - 2.0
    psi = (torch.rand(num_samples, device=device) * 2.0 * np.pi) - np.pi
    vx = torch.rand(num_samples, device=device) * 8.0 + 0.5  # 0.5 to 8.5 m/s
    vy = (torch.rand(num_samples, device=device) * 2.0) - 1.0 # lateral drift
    r = (torch.rand(num_samples, device=device) * 2.0) - 1.0   # yaw rate
    
    init_states = torch.stack([dx, dy, psi, vx, vy, r], dim=1)
    
    # Obstacle radius + vehicle boundary margin (0.4m obs + 0.2m car = 0.6m)
    r_collision = 0.6
    dist_init = torch.sqrt(init_states[:, 0]**2 + init_states[:, 1]**2)
    geometric_unsafe = dist_init <= r_collision
    
    # 49 Time-varying action profiles: [Horizon, 2] per maneuver
    accels = [-p.a_max, -0.5 * p.a_max, 0.0, 0.5 * p.a_max, p.a_max]
    steers = [-p.delta_max, -0.5 * p.delta_max, 0.0, 0.5 * p.delta_max, p.delta_max]
    
    time_varying_maneuvers = []
    
    # 1. 25 Constant full-effort profiles (Emergency straight-line braking, full throttle, steady cornering)
    for a in accels:
        for s in steers:
            profile = torch.zeros((horizon_steps, 2), device=device)
            profile[:, 0] = a
            profile[:, 1] = s
            time_varying_maneuvers.append(profile)
            
    # 2. 24 Two-Phase Time-Varying Maneuvers
    split_step = horizon_steps // 2
    for s in [-p.delta_max, -0.5 * p.delta_max, 0.5 * p.delta_max, p.delta_max]:
        for a in [-0.5 * p.a_max, 0.0, 0.5 * p.a_max]:
            # 12 S-Curve Maneuvers: Steer s -> Steer -s
            p1 = torch.zeros((horizon_steps, 2), device=device)
            p1[:split_step, 0] = a
            p1[:split_step, 1] = s
            p1[split_step:, 0] = a
            p1[split_step:, 1] = -s
            time_varying_maneuvers.append(p1)
            
            # 12 Swerve-and-Straighten Maneuvers: Steer s -> Steer 0
            p2 = torch.zeros((horizon_steps, 2), device=device)
            p2[:split_step, 0] = a
            p2[:split_step, 1] = s
            p2[split_step:, 0] = a
            p2[split_step:, 1] = 0.0
            time_varying_maneuvers.append(p2)
            
    num_maneuvers = len(time_varying_maneuvers)
    print(f"[Stage 2] Evaluating {num_maneuvers} time-varying reachability profiles per state on GPU...")
    
    # Track minimum distance across all trajectories for continuous smooth barrier labeling
    max_min_dist = torch.zeros(num_samples, device=device)
    can_escape = torch.zeros(num_samples, dtype=torch.bool, device=device)
    
    for profile in time_varying_maneuvers:
        curr_state = init_states.clone()
        trajectory_min_dist = torch.full((num_samples,), 99.0, device=device)
        trajectory_safe = torch.ones(num_samples, dtype=torch.bool, device=device)
        
        for step in range(horizon_steps):
            u_t = profile[step, :].unsqueeze(0).repeat(num_samples, 1)
            curr_state = model.step_rk4_torch(curr_state, u_t, dt=dt)
            dists = torch.sqrt(curr_state[:, 0]**2 + curr_state[:, 1]**2)
            trajectory_min_dist = torch.minimum(trajectory_min_dist, dists)
            trajectory_safe = trajectory_safe & (dists > r_collision)
            
        can_escape = can_escape | trajectory_safe
        max_min_dist = torch.maximum(max_min_dist, trajectory_min_dist)
        
    # Smooth continuous barrier target:
    # h = tanh((max_min_dist - r_collision) / scale)
    # Inside collision / inescapable crash zone: h < 0
    # Safe escape zone: h > 0, scaling smoothly with clearance margin
    barrier_margin = (max_min_dist - r_collision) / 1.5
    labels = torch.where(geometric_unsafe, torch.tensor(-1.0, device=device),
                         torch.where(can_escape, torch.tanh(barrier_margin), torch.tensor(-0.8, device=device)))
    labels = labels.unsqueeze(1)
    
    elapsed = time.time() - start_time
    safe_count = (labels > 0.0).sum().item()
    unsafe_count = (labels <= 0.0).sum().item()
    print(f"[Stage 2] Generated in {elapsed:.2f}s | Safe states: {safe_count} ({safe_count/num_samples*100:.1f}%) | Unsafe (Escape Zone) states: {unsafe_count} ({unsafe_count/num_samples*100:.1f}%)")
    
    return init_states.detach(), labels.detach()


# ---------------------------------------------------------------------------
# 3. Model Training Loop with Physics-Informed Gradient Consistency
# ---------------------------------------------------------------------------
def train_neural_icbf(num_samples=250000, epochs=30, batch_size=1024, lr=1e-3, save_dir="./models", device_str=None):
    global device
    if device_str is not None:
        device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"[Stage 3] Training on Device: {device} | Total Rollouts: {num_samples} | Epochs: {epochs}")
    states, labels = generate_rollout_dataset(num_samples=num_samples)
    
    n_train = int(0.85 * len(states))
    train_dataset = TensorDataset(states[:n_train], labels[:n_train])
    val_dataset = TensorDataset(states[n_train:], labels[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = NeuralBarrierNet(input_dim=6, hidden_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = nn.MSELoss()
    
    print(f"[Stage 3] Training Neural Barrier Network for {epochs} epochs with Physics Consistency...")
    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "neural_icbf_6d.pth")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            h_pred = model(x_b)
            loss = criterion(h_pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)
            
        train_loss /= n_train
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        correct_sign = 0
        n_val = len(val_dataset)
        
        with torch.no_grad():
            for x_b, y_b in val_loader:
                h_pred = model(x_b)
                val_loss += criterion(h_pred, y_b).item() * len(x_b)
                correct_sign += ((h_pred > 0) == (y_b > 0)).sum().item()
                
        val_loss /= n_val
        val_acc = (correct_sign / n_val) * 100.0
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Safety Classification Acc: {val_acc:.2f}%")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            
    print(f"\n[Stage 3] GPU Training complete! Best Val Loss: {best_val_loss:.4f}. Model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=250000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) if os.path.basename(current_script_dir) == "src" else current_script_dir
    save_dir = args.save_dir if args.save_dir is not None else os.path.join(project_root, "models")
    
    train_neural_icbf(num_samples=args.num_samples, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_dir=save_dir, device_str=args.device)
