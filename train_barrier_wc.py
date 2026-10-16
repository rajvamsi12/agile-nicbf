"""
train_barrier_wc.py  --  worst-case-mu, wall-aware, footprint-aware neural barrier.
 
Differences from train_neural_icbf.py (and why):
  1. Wall-aware.  The old net only saw obstacle-relative (dx,dy,...), so it could not know
     where the wall is.  Here the input includes the car's ABSOLUTE lateral position `py`,
     and the label counts wall violation (same disk test as the env's out_of_bounds).
  2. Footprint-aware.  Clearance is measured with the env's 3 disks (r=0.21 at -lr, 0, +lf),
     so no r_margin / 0.60/r_eff rescale hack is needed.  Obstacle radius is an input.
  3. Worst-case mu.  For every state the label is min over mu in MU_GRID of the best-escape
     clearance (rollouts use that mu, and ax is capped at mu*g like a real tire).
  4. Steering slew.  Rollouts obey |delta_dot| <= sv_max starting from delta0, which is an
     input (controller knows it: current_u[1]).  The old labels assumed instant full lock.
  5. Smooth label tanh(clearance/0.5) instead of the -0.8 / -1 jumps.
 
Feature vector (9D), build it with `features()` below in BOTH the shield and MPPI:
    [dx, dy, psi, vx, vy, r, py, r_obs, delta0]     dx,dy = car - obstacle
 
HONESTY NOTE: this is a *learned* escape-margin function over 49 open-loop primitives.
It is an inner approximation of the safe set and is NOT certified.  Report it as
"learned barrier enforced by a QP", not as a forward-invariance guarantee.
"""
import os, sys, time, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
 
here = os.path.dirname(os.path.abspath(__file__))
for d in (here, os.path.join(here, "src")):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)
from dynamic_bicycle_env import DynamicBicycleModel, DynamicBicycleParams
 
MU_GRID = [0.4, 0.55, 0.7, 0.85, 1.0, 1.1]
R_DISK, TRACK_W, SAFETY_BUF = 0.21, 4.0, 0.05
FEAT_DIM = 9
 
# sampling box (keep controller inputs inside it; features() clamps)
LO = torch.tensor([-8.0, -3.2, -1.2, 0.3, -1.5, -2.0, -1.79, 0.20, -0.4189])
HI = torch.tensor([ 3.0,  3.2,  1.2, 12.0, 1.5,  2.0,  1.79, 0.35,  0.4189])
 
 
class NeuralBarrierNetWC(nn.Module):
    """Same MLP shape as before but with input normalisation baked in."""
    def __init__(self, in_dim=FEAT_DIM, hidden=256):
        super().__init__()
        self.register_buffer("mid", ((LO + HI) / 2).clone())
        self.register_buffer("halfrange", ((HI - LO) / 2).clone())
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 128), nn.Tanh(),
                                 nn.Linear(128, 1), nn.Tanh())
 
    def forward(self, x):
        return self.net((x - self.mid) / self.halfrange)
 
 
def features(state, obs, delta0):
    """Torch features. state [...,6]=(px,py,psi,vx,vy,r) ; obs [...,3]=(ox,oy,ro) ; delta0 [...]."""
    dx = state[..., 0] - obs[..., 0]
    dy = state[..., 1] - obs[..., 1]
    f = torch.stack([dx, dy, state[..., 2], state[..., 3], state[..., 4], state[..., 5],
                     state[..., 1], obs[..., 2], delta0], dim=-1)
    return torch.max(torch.min(f, HI.to(f.device)), LO.to(f.device))
 
 
def feature_rates(fdot6, f_py_dot):
    """d/dt of the 9 features given the 6 state derivatives (obstacle static, r_obs const, delta0 ~ const)."""
    z = torch.zeros_like(f_py_dot)
    return torch.stack([fdot6[0], fdot6[1], fdot6[2], fdot6[3], fdot6[4], fdot6[5], f_py_dot, z, z], dim=-1)
 
 
def maneuvers(p, H, device):
    accels = [-p.a_max, -0.5 * p.a_max, 0.0, 0.5 * p.a_max, p.a_max]
    steers = [-p.delta_max, -0.5 * p.delta_max, 0.0, 0.5 * p.delta_max, p.delta_max]
    out, split = [], H // 2
    for a in accels:
        for s in steers:
            m = torch.zeros(H, 2, device=device); m[:, 0] = a; m[:, 1] = s; out.append(m)
    for s in [-p.delta_max, -0.5 * p.delta_max, 0.5 * p.delta_max, p.delta_max]:
        for a in [-0.5 * p.a_max, 0.0, 0.5 * p.a_max]:
            m = torch.zeros(H, 2, device=device); m[:, 0] = a; m[:split, 1] = s; m[split:, 1] = -s; out.append(m)
            m = torch.zeros(H, 2, device=device); m[:, 0] = a; m[:split, 1] = s; m[split:, 1] = 0.0; out.append(m)
    return out  # 49
 
 
@torch.no_grad()
def worst_case_labels(feats, H=40, dt=0.02, mu_grid=MU_GRID, device="cpu", chunk=100000):
    """feats [N,9] -> (labels [N,1], margin [N,mu]) ; margin in metres, min over mu of max over maneuvers of min over time."""
    N = feats.shape[0]
    out_margin = torch.empty(N, len(mu_grid), device=device)
    for lo in range(0, N, chunk):
        f = feats[lo:lo + chunk]
        n = f.shape[0]
        s0 = torch.stack([f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4], f[:, 5]], 1)
        # y absolute is f[:,6]; obstacle at origin so the rollout's py must be ABSOLUTE for the wall test.
        # We therefore keep relative px,py for the obstacle test and track an absolute-y offset for the wall.
        y_off = f[:, 6] - f[:, 1]            # = oy
        r_obs, d0 = f[:, 7], f[:, 8]
        for mi, mu in enumerate(mu_grid):
            params = DynamicBicycleParams(); params.mu = float(mu)
            model = DynamicBicycleModel(params=params, device=device)
            p = model.p
            ax_cap = min(p.a_max, mu * p.g)
            best = torch.full((n,), -1e3, device=device)
            for prof in maneuvers(p, H, device):
                st, d_act = s0.clone(), d0.clone()
                worst = torch.full((n,), 1e3, device=device)
                for k in range(H):
                    cmd_d = prof[k, 1].expand(n)
                    d_act = d_act + torch.clamp(cmd_d - d_act, -p.sv_max * dt, p.sv_max * dt)
                    ax = prof[k, 0].clamp(-ax_cap, ax_cap).expand(n)
                    st = model.step_rk4_torch(st, torch.stack([ax, d_act], 1), dt=dt)
                    # wall test needs absolute y: shift py by oy for the wall, keep relative for obstacle
                    c, sn = torch.cos(st[:, 2]), torch.sin(st[:, 2])
                    cl = torch.full((n,), 1e3, device=device)
                    for l in (-p.lr, 0.0, p.lf):
                        cx, cy = st[:, 0] + l * c, st[:, 1] + l * sn
                        d_o = torch.sqrt(cx ** 2 + cy ** 2 + 1e-8) - r_obs - R_DISK
                        d_w = TRACK_W / 2 - ((cy + y_off).abs() + R_DISK)
                        cl = torch.minimum(cl, torch.minimum(d_o, d_w))
                    worst = torch.minimum(worst, cl)
                best = torch.maximum(best, worst)
            out_margin[lo:lo + n, mi] = best
    wc = out_margin.min(dim=1).values - SAFETY_BUF
    labels = torch.tanh(wc / 0.5).unsqueeze(1)
    return labels, out_margin
 
 
def sample_states(N, device):
    u = torch.rand(N, FEAT_DIM, device=device)
    lo, hi = LO.to(device), HI.to(device)
    f = lo + u * (hi - lo)
    # py = oy + dy must be inside the track; sample oy, py then derive dy
    oy = (torch.rand(N, device=device) * 2 - 1) * 1.4
    py = (torch.rand(N, device=device) * 2 - 1) * 1.79
    f[:, 6], f[:, 1] = py, py - oy
    # bias dx toward the interesting region just ahead of the obstacle
    near = torch.rand(N, device=device) < 0.5
    f[near, 0] = -(torch.rand(int(near.sum()), device=device) * 4.0) + 0.5
    return f
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=250000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=40, help="steps of 0.02 s (40 = 0.8 s)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="models/neural_icbf_9d_wc.pth")
    ap.add_argument("--mu_grid", type=float, nargs="+", default=MU_GRID)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(0)
 
    t0 = time.time()
    X = sample_states(a.num_samples, dev)
    Y, M = worst_case_labels(X, H=a.horizon, mu_grid=a.mu_grid, device=dev)
    print(f"labels in {time.time()-t0:.1f}s | safe(wc) {(Y>0).float().mean()*100:.1f}%")
    for i, mu in enumerate(a.mu_grid):
        print(f"  mu={mu:.2f}: escapable {(M[:,i]-SAFETY_BUF>0).float().mean()*100:5.1f}%")
    print("  (escapable % must be non-increasing as mu drops; worst-case <= every row)")
 
    n_tr = int(0.85 * len(X))
    tr = DataLoader(TensorDataset(X[:n_tr], Y[:n_tr]), batch_size=1024, shuffle=True)
    Xv, Yv = X[n_tr:], Y[n_tr:]
    net = NeuralBarrierNetWC().to(dev)
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best = 1e9
    os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    for ep in range(1, a.epochs + 1):
        net.train()
        for xb, yb in tr:
            opt.zero_grad(); loss = nn.functional.mse_loss(net(xb), yb); loss.backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xv); vl = nn.functional.mse_loss(pv, Yv).item()
            acc = ((pv > 0) == (Yv > 0)).float().mean().item() * 100
            false_safe = ((pv > 0) & (Yv <= 0)).float().sum() / max(1, (Yv <= 0).sum())
        if vl < best:
            best = vl; torch.save(net.state_dict(), a.save)
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:02d} val_mse {vl:.4f} sign_acc {acc:.2f}% false-safe {false_safe.item()*100:.2f}%")
    print(f"saved {a.save}  (best val mse {best:.4f})")
    print("Report the FALSE-SAFE rate (predicts h>0 where label<=0): that is the dangerous error.")
 
 
if __name__ == "__main__":
    main()
