"""
train_terminal_barrier.py -- Neural terminal safety certificate for the predictive safety filter.

h_theta(x) > 0  <=>  from state x there EXISTS a backup maneuver that brings the car to a stop without
touching the obstacle (worst case over a small friction grid), simulated with the SAME actuator model as
the env and the filter (steering slew sv_max, first-order motor lag tau).

Differences from retrain_neural_icbf_robust.py (whose network extrapolates in slides):
  * state range covers real racing transients: |r| <= 6 rad/s, |vy| <= 3.5 m/s, vx in [0, 10] m/s
  * labels use actuator dynamics (no instantaneous steering) and roll out to a STOP (1.5 s), not 0.6 s
  * inputs: [dx, dy, psi, vx, vy, r, delta_actual, r_obs] in metres/radians (no ad-hoc scaling);
    delta_actual matters because the servo needs time to unwind a full-lock steering angle
  * label = tanh((best-maneuver min clearance - collision radius) / 0.3): signed, continuous

Output: models/terminal_barrier_8d.pth (+ .json with validation metrics). Does not touch existing models.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "src")]
from dynamic_bicycle_env import DynamicBicycleModel

R_DISK = 0.21


class TerminalBarrierNet(nn.Module):
    """8-D input [dx, dy, psi, vx, vy, r, delta, r_obs] -> h in (-1, 1)."""
    def __init__(self, input_dim=8, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 128), nn.Tanh(), nn.Linear(128, 1), nn.Tanh())
        # Fixed input normalisation (keeps the checkpoint self-contained)
        self.register_buffer("scale", torch.tensor([5.0, 2.5, np.pi, 10.0, 3.5, 6.0, 0.42, 0.35]))

    def forward(self, x):
        return self.net(x / self.scale)


def backup_policies(a_max):
    """(brake decel, steering policy) pairs. Steering policy: ('hold'|'zero'|'yaw'|const angle)."""
    pols = []
    for brake in (2.0, 4.5, 8.0):
        for steer in ("hold", "zero", "yaw", 0.42, -0.42, 0.21, -0.21):
            pols.append((brake, steer))
    for acc in (0.0, 3.0):            # swerve without braking (escape sideways), then brake later
        for steer in (0.42, -0.42, 0.21, -0.21):
            pols.append((-acc, steer))
    return pols


@torch.no_grad()
def label_states(X, mu_grid, steps, dt, device, sv_max=3.2, tau=0.05, yaw_gain=0.1, chunk=60000):
    """X [N, 8] -> best min clearance over backup policies, worst case over mu. Obstacle at origin."""
    N = X.shape[0]
    out = torch.empty(N, device=device)
    for c0 in range(0, N, chunk):
        Xc = X[c0:c0 + chunk]
        worst = torch.full((Xc.shape[0],), 1e9, device=device)
        for mu in mu_grid:
            model = DynamicBicycleModel(device=device)
            model.p.mu = float(mu)
            p = model.p
            a_lim = min(p.a_max, p.mu * p.g)
            best = torch.full((Xc.shape[0],), -1e9, device=device)
            for brake, steer in backup_policies(p.a_max):
                x = Xc[:, :6].clone()
                a_d = Xc[:, 6].clone()
                a_ax = torch.zeros_like(a_d)
                d0 = a_d.clone()
                r_coll = Xc[:, 7] + R_DISK
                mind = torch.full_like(a_d, 1e9)
                for t in range(steps):
                    moving = x[:, 3] > 0.3
                    c_ax = torch.where(moving, torch.full_like(a_d, -brake), torch.zeros_like(a_d)).clamp(-a_lim, a_lim)
                    if steer == "hold":
                        c_d = d0
                    elif steer == "zero":
                        c_d = torch.zeros_like(a_d)
                    elif steer == "yaw":
                        c_d = (-yaw_gain * x[:, 5]).clamp(-p.delta_max, p.delta_max)
                    else:
                        c_d = torch.full_like(a_d, float(steer))
                    a_d = a_d + (c_d - a_d).clamp(-sv_max * dt, sv_max * dt)
                    a_ax = (a_ax + dt / tau * (c_ax - a_ax)).clamp(-a_lim, a_lim)
                    x_new = model.step_rk4_torch(x, torch.stack([a_ax, a_d], 1), dt=dt)
                    x = torch.where(moving.unsqueeze(1) | (c_ax > 0).unsqueeze(1), x_new, x)  # stopped stays put
                    c, s = torch.cos(x[:, 2]), torch.sin(x[:, 2])
                    d = torch.minimum(torch.hypot(x[:, 0], x[:, 1]),
                                      torch.minimum(torch.hypot(x[:, 0] - p.lr * c, x[:, 1] - p.lr * s),
                                                    torch.hypot(x[:, 0] + p.lf * c, x[:, 1] + p.lf * s)))
                    mind = torch.minimum(mind, d - r_coll)
                best = torch.maximum(best, mind)
            worst = torch.minimum(worst, best)
        out[c0:c0 + chunk] = worst
    return out


def sample_states(n, device, g, near_frac=0.7):
    """Mixture: near_frac of positions within 3 m of the obstacle (where the decision boundary is), rest uniform."""
    u = lambda lo, hi: torch.rand(n, device=device, generator=g) * (hi - lo) + lo
    X = torch.stack([u(-5, 5), u(-2.5, 2.5), u(-np.pi, np.pi), u(0.0, 10.0), u(-3.5, 3.5), u(-6, 6),
                     u(-0.42, 0.42), u(0.20, 0.35)], dim=1)
    near = torch.rand(n, device=device, generator=g) < near_frac
    rho, th = u(0.4, 3.0), u(-np.pi, np.pi)
    X[near, 0] = (rho * torch.cos(th))[near]
    X[near, 1] = (rho * torch.sin(th))[near].clamp(-2.5, 2.5)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=75)        # 1.5 s: stops from ~7 m/s at 4.5 m/s^2
    ap.add_argument("--mu", type=float, nargs="+", default=[1.0, 1.0489])
    ap.add_argument("--out", default=os.path.join(here, "models", "terminal_barrier_8d.pth"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=dev); g.manual_seed(0)
    torch.manual_seed(0)

    t0 = time.time()
    X = sample_states(a.n, dev, g)
    clr = label_states(X, a.mu, a.steps, 0.02, dev)
    y = torch.tanh(clr / 0.3).unsqueeze(1)
    print(f"labels: {a.n} states in {time.time()-t0:.0f}s | safe {(y > 0).float().mean().item()*100:.1f}%", flush=True)

    n_tr = int(0.85 * a.n)
    net = TerminalBarrierNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    best, metrics = 1e9, {}
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(n_tr, device=dev)
        for i in range(0, n_tr, 1024):
            idx = perm[i:i + 1024]
            w = 1.0 + 4.0 * (y[idx] <= 0).float()           # false "safe" is the costly error for a certificate
            loss = (w * (net(X[idx]) - y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            pv, yv = net(X[n_tr:]), y[n_tr:]
            vl = nn.functional.mse_loss(pv, yv).item()
            acc = ((pv > 0) == (yv > 0)).float().mean().item()
            false_safe = ((pv > 0) & (yv <= 0)).float().sum().item() / max(1, (yv <= 0).sum().item())
            false_safe_m = ((pv > 0.1) & (yv <= 0)).float().sum().item() / max(1, (yv <= 0).sum().item())
        if vl < best:
            best = vl
            torch.save(net.state_dict(), a.out)
            metrics = {"epoch": ep, "val_mse": vl, "val_sign_acc": acc, "false_safe_rate@0": false_safe,
                       "false_safe_rate@0.1": false_safe_m}
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:2d} val_mse {vl:.4f} sign_acc {acc*100:.2f}% false_safe@0 {false_safe*100:.2f}% "
                  f"@0.1 {false_safe_m*100:.2f}%", flush=True)
    metrics.update({"n": a.n, "steps": a.steps, "mu_grid": a.mu, "train_time_s": time.time() - t0})
    json.dump(metrics, open(a.out.replace(".pth", ".json"), "w"), indent=2)
    print("saved", a.out, metrics)


if __name__ == "__main__":
    main()
