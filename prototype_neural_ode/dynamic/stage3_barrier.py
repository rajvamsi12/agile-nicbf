"""
Stage 3 -- Learned state barriers h_theta (paper Sec. 4.3 / 5.2, Algorithm 1 line 27: labels generated through F_theta).

Viability labels: a state is safe iff SOME backup maneuver (brake levels x steering policies, plus swerves), simulated
through the LEARNED dynamics F_theta with the actuator model (steering slew, motor lag; servo starts centred), stays
clear of the constraint until it stops (1.5 s). label = tanh(best-maneuver min clearance / 0.3).
  python stage3_barrier.py obs    -> models/h_obs.pth
  python stage3_barrier.py wall   -> models/h_wall.pth
"""
import time, json, argparse
import numpy as np
import torch
import torch.nn as nn
from common import FNet, HObsNet, HWallNet, load, save, R_DISK, TRACK_HALF, SV_MAX, TAU, DT, plant_params

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["obs", "wall"])
ap.add_argument("--n", type=int, default=200000)
ap.add_argument("--steps", type=int, default=75)
ap.add_argument("--epochs", type=int, default=60)
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
F = load(FNet, "F_theta.pth", dev)
for prm in F.parameters():
    prm.requires_grad_(False)
P = plant_params()
LF, LR = P.lf, P.lr
A_LIM = min(P.a_max, P.mu * P.g)


def maneuvers():
    m = []
    for brake in (2.0, 4.5, 8.0):
        for steer in ("zero", "yaw", 0.42, -0.42, 0.21, -0.21):
            m.append((brake, steer))
    for steer in (0.42, -0.42, 0.21, -0.21):
        m.append((0.0, steer))
    return m


def disks(x):
    c, s = torch.cos(x[:, 2]), torch.sin(x[:, 2])
    return [(x[:, 0] - LR * c, x[:, 1] - LR * s), (x[:, 0], x[:, 1]), (x[:, 0] + LF * c, x[:, 1] + LF * s)]


@torch.no_grad()
def clearance(x, kind, extra):
    """x [N,6] absolute-frame state with the obstacle at the origin / the top wall at y = TRACK_HALF."""
    if kind == "obs":
        return torch.stack([torch.hypot(px, py) for px, py in disks(x)], 0).min(0).values - (extra + R_DISK)
    return torch.stack([TRACK_HALF - (py + R_DISK) for _, py in disks(x)], 0).min(0).values


@torch.no_grad()
def label(x0, kind, extra, chunk=50000):
    out = torch.empty(x0.shape[0], device=dev)
    for c0 in range(0, x0.shape[0], chunk):
        xc, ec = x0[c0:c0 + chunk], (extra[c0:c0 + chunk] if extra is not None else None)
        best = torch.full((xc.shape[0],), -1e9, device=dev)
        for brake, steer in maneuvers():
            x = xc.clone(); ad = torch.zeros(xc.shape[0], device=dev); aa = torch.zeros_like(ad)
            mind = clearance(x, kind, ec)
            for t in range(a.steps):
                moving = x[:, 3] > 0.3
                c_ax = torch.where(moving, torch.full_like(ad, -brake), torch.zeros_like(ad))
                if steer == "zero":
                    c_d = torch.zeros_like(ad)
                elif steer == "yaw":
                    c_d = (-0.1 * x[:, 5]).clamp(-0.42, 0.42)
                else:
                    c_d = torch.full_like(ad, float(steer))
                ad = ad + (c_d - ad).clamp(-SV_MAX * DT, SV_MAX * DT)
                aa = (aa + DT / TAU * (c_ax - aa)).clamp(-A_LIM, A_LIM)
                xn = F.step(x, torch.stack([aa, ad], 1))
                x = torch.where((moving | (c_ax > 0)).unsqueeze(1), xn, x)
                mind = torch.minimum(mind, clearance(x, kind, ec))
            best = torch.maximum(best, mind)
        out[c0:c0 + chunk] = best
    return out


def sample(n, kind):
    u = lambda lo, hi: torch.rand(n, device=dev) * (hi - lo) + lo
    psi, vx, vy, r = u(-np.pi, np.pi), u(0.0, 5.5), u(-1.5, 1.5), u(-3.0, 3.0)
    if kind == "obs":
        dx, dy = u(-6.0, 3.0), u(-2.5, 2.5)
        near = torch.rand(n, device=dev) < 0.7
        rho, th = u(0.3, 3.5), u(-np.pi, np.pi)
        dx = torch.where(near, rho * torch.cos(th), dx); dy = torch.where(near, rho * torch.sin(th), dy)
        ro = u(0.20, 0.35)
        x = torch.stack([dx, dy, psi, vx, vy, r], 1)
        return x, ro, torch.cat([x, ro.unsqueeze(1)], 1)
    d = torch.where(torch.rand(n, device=dev) < 0.7, u(0.2, 1.6), u(0.2, 4.0))   # distance of CG to the top wall
    x = torch.stack([torch.zeros(n, device=dev), TRACK_HALF - d, psi, vx, vy, r], 1)
    return x, None, torch.stack([d, psi, vx, vy, r], 1)


t0 = time.time()
x0, extra, feats = sample(a.n, a.kind)
clr = label(x0, a.kind, extra)
y = torch.tanh(clr / 0.3)
print(f"[{a.kind}] labelled {a.n} states through F_theta in {time.time()-t0:.0f}s | safe {(y > 0).float().mean().item()*100:.1f}%",
      flush=True)

net = (HObsNet() if a.kind == "obs" else HWallNet()).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
ntr = int(0.85 * a.n)
best, meta = 1e9, {}
for ep in range(1, a.epochs + 1):
    net.train()
    perm = torch.randperm(ntr, device=dev)
    for i in range(0, ntr, 1024):
        j = perm[i:i + 1024]
        w = 1.0 + 4.0 * (y[j] <= 0).float()
        loss = (w * (net(feats[j]) - y[j]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
    net.eval()
    with torch.no_grad():
        pv, yv = net(feats[ntr:]), y[ntr:]
        vl = nn.functional.mse_loss(pv, yv).item()
        uns = yv <= 0
        fs = {th: ((pv >= th) & uns).float().sum().item() / max(1, uns.sum().item()) for th in (0.0, 0.1, 0.2)}
        acc = ((pv > 0) == (yv > 0)).float().mean().item()
    if vl < best:
        best = vl
        meta = {"epoch": ep, "val_mse": vl, "sign_acc": acc, "false_safe@0": fs[0.0], "false_safe@0.1": fs[0.1],
                "false_safe@0.2": fs[0.2], "safe_frac": (y > 0).float().mean().item(), "n": a.n}
        save(net, f"h_{a.kind}.pth", meta)
    if ep % 10 == 0 or ep == 1:
        print(f"[{a.kind}] epoch {ep:2d} val_mse {vl:.4f} acc {acc*100:.2f}% false_safe@0 {fs[0.0]*100:.2f}% "
              f"@0.2 {fs[0.2]*100:.2f}%", flush=True)
meta["total_time_s"] = time.time() - t0
save(load(HObsNet if a.kind == "obs" else HWallNet, f"h_{a.kind}.pth", dev), f"h_{a.kind}.pth", meta)
print(f"[{a.kind}] best:", json.dumps(meta), flush=True)
