"""
common.py -- Shared definitions for the Neural ICBF pipeline adapted to the dynamic bicycle (paper Algorithm 1).

  F_theta : Neural ODE  x_dot = F_theta(x, u)        x = [px, py, psi, vx, vy, r], u = [ax, delta] (ACTUAL actuators)
  pi_theta: imitation policy u_nom = pi_theta(x)     (track following, obstacle-unaware, as in the paper)
  h_obs   : learned state barrier vs one obstacle    h(dx, dy, psi, vx, vy, r, r_obs)
  h_wall  : learned state barrier vs the top wall    h(d_wall, psi, vx, vy, r); bottom wall by mirror symmetry

The control input u is the actuator state, so the paper's integral law u_dot = phi + v is exactly the actuator
rate: steering slew |delta_dot| <= sv_max, and the first-order motor lag realises any a_dot via
ax_cmd = ax + tau * a_dot.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS = os.environ.get("NICBF_MODELS", os.path.join(HERE, "models"))
os.makedirs(MODELS, exist_ok=True)
for _d in (REPO, os.path.join(REPO, "src")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dynamic_bicycle_env import DynamicBicycleModel  # noqa: E402  (ground-truth plant, used only to make data)

PLANT_MU = 1.0          # friction of the evaluation plant (stress_grid sets env.model.p.mu = mu = 1.0)
TRACK_HALF = 2.0
R_DISK = 0.21
DT = 0.02
SV_MAX, TAU = 3.2, 0.05


def plant_params(mu=PLANT_MU):
    m = DynamicBicycleModel(device="cpu")
    m.p.mu = float(mu)
    return m.p


def mlp(i, o, h=256, n=3, act=nn.Tanh, out_act=None):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class FNet(nn.Module):
    """Neural ODE vector field. Translation-invariant in (px, py); heading enters as (cos, sin).
    Smooth (tanh) activations: the ICBF needs d F / d u and second derivatives of h along F."""
    def __init__(self):
        super().__init__()
        self.net = mlp(7, 6, h=256, n=3)
        self.register_buffer("in_scale", torch.tensor([1, 1, 6.0, 2.0, 4.0, 9.5, 0.42]))
        self.register_buffer("out_scale", torch.ones(6))

    def forward(self, x, u):
        feat = torch.stack([torch.cos(x[..., 2]), torch.sin(x[..., 2]), x[..., 3], x[..., 4], x[..., 5],
                            u[..., 0], u[..., 1]], dim=-1) / self.in_scale
        return self.net(feat) * self.out_scale

    def step(self, x, u, dt=DT):
        """RK4 with the plant's standstill handling (vx >= 0, vy = r = 0 when stopped)."""
        k1 = self(x, u); k2 = self(x + 0.5 * dt * k1, u); k3 = self(x + 0.5 * dt * k2, u); k4 = self(x + dt * k3, u)
        n = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        vx = n[..., 3].clamp(min=0.0)
        stop = (vx <= 1e-4).to(n.dtype)
        return torch.cat([n[..., :3], vx.unsqueeze(-1), n[..., 4:6] * (1 - stop).unsqueeze(-1)], dim=-1)


class PiNet(nn.Module):
    """Imitation policy: (py, cos psi, sin psi, vx, vy, r, v_target) -> desired actuator values (ax, delta)."""
    def __init__(self):
        super().__init__()
        self.net = mlp(7, 2, h=256, n=4)
        self.register_buffer("in_scale", torch.tensor([2.0, 1, 1, 6.0, 2.0, 4.0, 6.0]))
        self.register_buffer("out_scale", torch.tensor([9.5, 0.42]))

    def forward(self, x, v_target):
        vt = torch.as_tensor(v_target, dtype=x.dtype, device=x.device).expand(x.shape[:-1])
        feat = torch.stack([x[..., 1], torch.cos(x[..., 2]), torch.sin(x[..., 2]), x[..., 3], x[..., 4], x[..., 5], vt],
                           dim=-1) / self.in_scale
        return torch.tanh(self.net(feat)) * self.out_scale


class HObsNet(nn.Module):
    """State barrier vs one obstacle: (dx, dy, psi, vx, vy, r, r_obs) -> h in (-1, 1); dx = px - ox."""
    def __init__(self):
        super().__init__()
        self.net = mlp(8, 1, h=256, n=3, out_act=nn.Tanh)
        self.register_buffer("in_scale", torch.tensor([4.0, 2.5, 1, 1, 6.0, 2.0, 4.0, 0.35]))

    def forward(self, s):
        feat = torch.stack([s[..., 0], s[..., 1], torch.cos(s[..., 2]), torch.sin(s[..., 2]), s[..., 3], s[..., 4],
                            s[..., 5], s[..., 6]], dim=-1) / self.in_scale
        return self.net(feat).squeeze(-1)


class HWallNet(nn.Module):
    """State barrier vs the top wall: (d = TRACK_HALF - py, psi, vx, vy, r) -> h in (-1, 1)."""
    def __init__(self):
        super().__init__()
        self.net = mlp(6, 1, h=256, n=3, out_act=nn.Tanh)
        self.register_buffer("in_scale", torch.tensor([2.0, 1, 1, 6.0, 2.0, 4.0]))

    def forward(self, s):
        feat = torch.stack([s[..., 0], torch.cos(s[..., 1]), torch.sin(s[..., 1]), s[..., 2], s[..., 3], s[..., 4]],
                           dim=-1) / self.in_scale
        return self.net(feat).squeeze(-1)


def save(model, name, meta=None):
    path = os.path.join(MODELS, name)
    torch.save(model.state_dict(), path)
    if meta is not None:
        import json
        json.dump(meta, open(path.replace(".pth", ".json"), "w"), indent=2)
    return path


def load(cls, name, device="cpu"):
    m = cls().to(device)
    m.load_state_dict(torch.load(os.path.join(MODELS, name), map_location=device))
    m.eval()
    return m
