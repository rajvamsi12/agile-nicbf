import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

try:
    from .dynamic_bicycle_env import DynamicBicycleModel, DynamicBicycleParams
except ImportError:
    from dynamic_bicycle_env import DynamicBicycleModel, DynamicBicycleParams

class DRLRacingActor(nn.Module):
    """
    Deep RL Actor Network Architecture for Autonomous Racing.
    Architecture: MLP with 3 hidden layers (128-128-64 units).
    Input: [6D vehicle state, 5 closest obstacle relative coordinates (dx, dy, r)] -> 6 + 15 = 21 dim
    Output: [ax, delta] normalized actions in [-1, 1]
    """
    def __init__(self, input_dim=21, hidden_dim=128):
        super(DRLRacingActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        
    def forward(self, obs):
        return self.net(obs)

class DRLRacingPolicy:
    """
    Untrained Zero-Shot Neural Policy Baseline (Orthogonally Initialized).
    Placeholder baseline architecture; not trained against an environment reward function.
    """
    def __init__(self, v_target=6.0, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.p = DynamicBicycleParams()
        self.v_target = v_target
        self.actor = DRLRacingActor().to(self.device)
        self.actor.eval()
        
        # Standard orthogonal weight initialization (untrained baseline)
        with torch.no_grad():
            for m in self.actor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)
                    
    def build_observation(self, state, obstacles):
        """Constructs 21-dimensional state + obstacle feature vector."""
        px, py, psi, vx, vy, r = state
        features = [py / 2.0, psi / np.pi, vx / 10.0, vy / 2.0, r / 3.0, (self.v_target - vx) / 5.0]
        
        # 5 nearest obstacles
        obs_features = []
        if obstacles is not None and len(obstacles) > 0:
            dists = np.sqrt((obstacles[:, 0] - px)**2 + (obstacles[:, 1] - py)**2)
            idx_sorted = np.argsort(dists)[:5]
            for idx in idx_sorted:
                ox, oy, ro = obstacles[idx]
                dx = (ox - px) * np.cos(psi) + (oy - py) * np.sin(psi)
                dy = -(ox - px) * np.sin(psi) + (oy - py) * np.cos(psi)
                obs_features.extend([dx / 10.0, dy / 2.0, ro])
                
        # Pad if fewer than 5 obstacles
        while len(obs_features) < 15:
            obs_features.extend([2.0, 0.0, 0.0])
            
        full_obs = np.array(features + obs_features, dtype=np.float32)
        return torch.as_tensor(full_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def plan(self, state, obstacles):
        t0 = time.time()
        obs_tensor = self.build_observation(state, obstacles)
        with torch.no_grad():
            act_norm = self.actor(obs_tensor).squeeze().cpu().numpy()
            
        # Rescale normalized actions to physical bounds
        ax = float(act_norm[0] * self.p.a_max)
        delta = float(act_norm[1] * self.p.delta_max)
        
        latency = (time.time() - t0) * 1000.0
        return np.array([ax, delta], dtype=np.float32), {"latency_ms": latency}
