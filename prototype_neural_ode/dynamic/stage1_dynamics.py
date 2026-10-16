"""
Stage 1 -- Neural ODE F_theta for the 6-D dynamic bicycle (paper Sec. 4.1, eq. 7).

Data: state/actuator samples from the ground-truth plant (mu = 1.0), used ONLY to generate training data.
Loss: normalised derivative regression + K-step RK4 rollout loss (eq. 7).
Validation: derivative error, 1 s open-loop rollout error, and the accuracy of dF/du (autograd) that the ICBF uses.
Output: dynamic/models/F_theta.pth (+ .json metrics)
"""
import time, argparse, json
import numpy as np
import torch
from common import FNet, DynamicBicycleModel, PLANT_MU, save

ap = argparse.ArgumentParser()
ap.add_argument("--n_pairs", type=int, default=400000)
ap.add_argument("--n_traj", type=int, default=40000)
ap.add_argument("--k_roll", type=int, default=10)
ap.add_argument("--epochs", type=int, default=60)
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
truth = DynamicBicycleModel(device=dev); truth.p.mu = PLANT_MU


def sample(n):
    u = lambda lo, hi: torch.rand(n, device=dev) * (hi - lo) + lo
    x = torch.stack([torch.zeros(n, device=dev), u(-2, 2), u(-np.pi, np.pi), u(0.0, 6.5), u(-2.0, 2.0), u(-4.0, 4.0)], 1)
    uu = torch.stack([u(-9.5, 9.5), u(-0.42, 0.42)], 1)
    return x, uu


t0 = time.time()
X, U = sample(a.n_pairs)
with torch.no_grad():
    Y = truth.continuous_dynamics_torch(X, U)
# Trajectories: random start, piecewise-constant random actuator values held for the whole segment (+ jitter)
X0, U0 = sample(a.n_traj)
with torch.no_grad():
    Ujit = U0.unsqueeze(1) + torch.randn(a.n_traj, a.k_roll, 2, device=dev) * torch.tensor([1.0, 0.03], device=dev)
    Ujit[..., 0].clamp_(-9.5, 9.5); Ujit[..., 1].clamp_(-0.42, 0.42)
    traj = [X0]
    for k in range(a.k_roll):
        traj.append(truth.step_rk4_torch(traj[-1], Ujit[:, k], 0.02))
    traj = torch.stack(traj, 1)                                  # [n, k+1, 6]
print(f"data: {a.n_pairs} pairs + {a.n_traj} x {a.k_roll}-step trajectories in {time.time()-t0:.0f}s", flush=True)

net = FNet().to(dev)
net.out_scale.copy_(Y.std(0).clamp(min=1e-3))
w = 1.0 / net.out_scale
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
n_tr, nt_tr = int(0.9 * a.n_pairs), int(0.9 * a.n_traj)
B, BT = 4096, 512
for ep in range(1, a.epochs + 1):
    perm, permt = torch.randperm(n_tr, device=dev), torch.randperm(nt_tr, device=dev)
    nb = n_tr // B
    for i in range(nb):
        idx = perm[i * B:(i + 1) * B]
        l_der = (((net(X[idx], U[idx]) - Y[idx]) * w) ** 2).mean()
        j = permt[(i * BT) % nt_tr:(i * BT) % nt_tr + BT]
        x = traj[j, 0]; l_roll = 0.0
        for k in range(a.k_roll):
            x = net.step(x, Ujit[j, k])
            l_roll = l_roll + (((x - traj[j, k + 1])[:, 1:] / torch.tensor([0.05, 0.05, 0.1, 0.1, 0.2], device=dev)) ** 2).mean()
        loss = l_der + 0.1 * l_roll / a.k_roll
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
    if ep % 10 == 0 or ep == 1:
        with torch.no_grad():
            vl = (((net(X[n_tr:], U[n_tr:]) - Y[n_tr:]) * w) ** 2).mean().item()
        print(f"epoch {ep:3d} train {loss.item():.5f} val_deriv_nmse {vl:.5f}", flush=True)

# ---------------- validation ----------------
net.eval()
with torch.no_grad():
    vd = (((net(X[n_tr:], U[n_tr:]) - Y[n_tr:]) * w) ** 2).mean(0).cpu().numpy()
# 1 s open-loop rollouts in the operating regime (vx 3-4.5, moderate slip) with smooth random controls
g = torch.Generator(device=dev); g.manual_seed(1)
n = 2000
x0 = torch.stack([torch.zeros(n, device=dev), (torch.rand(n, device=dev, generator=g) - .5) * 2,
                  (torch.rand(n, device=dev, generator=g) - .5) * 0.8, 3 + 1.5 * torch.rand(n, device=dev, generator=g),
                  (torch.rand(n, device=dev, generator=g) - .5) * 0.6, (torch.rand(n, device=dev, generator=g) - .5) * 2], 1)
uc = torch.stack([(torch.rand(n, device=dev, generator=g) - .5) * 6, (torch.rand(n, device=dev, generator=g) - .5) * 0.5], 1)
with torch.no_grad():
    xa, xb = x0.clone(), x0.clone()
    for k in range(50):
        xa = truth.step_rk4_torch(xa, uc, 0.02); xb = net.step(xb, uc)
    pos_err = torch.hypot(xa[:, 0] - xb[:, 0], xa[:, 1] - xb[:, 1])
    yaw_err = (xa[:, 2] - xb[:, 2]).abs()
# dF/du accuracy on operating-regime states
xs = x0[:500].clone(); us = uc[:500].clone().requires_grad_(True)
Jt = torch.autograd.functional.jacobian(lambda uu: truth.continuous_dynamics_torch(xs, uu).sum(0), us)  # [6, n, 2]
Jn = torch.autograd.functional.jacobian(lambda uu: net(xs, uu).sum(0), us)
Jt, Jn = Jt.permute(1, 0, 2), Jn.permute(1, 0, 2)                                                       # [n, 6, 2]
rel = (Jn - Jt).norm(dim=(1, 2)) / Jt.norm(dim=(1, 2)).clamp(min=1e-6)
cos = torch.nn.functional.cosine_similarity(Jn.reshape(500, -1), Jt.reshape(500, -1), dim=1)
metrics = {
    "val_deriv_nmse_per_output": vd.tolist(),
    "rollout_1s_pos_err_m": {"median": pos_err.median().item(), "p95": pos_err.quantile(0.95).item()},
    "rollout_1s_yaw_err_rad": {"median": yaw_err.median().item(), "p95": yaw_err.quantile(0.95).item()},
    "dFdu_rel_err": {"median": rel.median().item(), "p95": rel.quantile(0.95).item()},
    "dFdu_cosine": {"median": cos.median().item(), "p5": cos.quantile(0.05).item()},
    "train_time_s": time.time() - t0,
}
print(json.dumps(metrics, indent=1), flush=True)
print("saved", save(net, "F_theta.pth", metrics))
