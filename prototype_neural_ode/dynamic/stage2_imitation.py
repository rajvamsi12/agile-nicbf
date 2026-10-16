"""
Stage 2 -- Imitation policy pi_theta (paper Sec. 4.2) with the GPU MPPI planner as the expert.

As in the paper, the nominal controller is obstacle-unaware: the expert does track following (centerline + target
speed) on an EMPTY track; obstacles and walls are handled only by the Neural ICBF filter.
  collect: python stage2_imitation.py collect --shard i/n      -> models/il_data_<i>.npz
  train:   python stage2_imitation.py train                    -> models/pi_theta.pth (+ .json)
"""
import os, sys, time, json, argparse, glob
import numpy as np
import torch
from common import PiNet, MODELS, PLANT_MU, REPO, save

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["collect", "train"])
ap.add_argument("--shard", default="0/1")
ap.add_argument("--episodes", type=int, default=360)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--K", type=int, default=1000)
ap.add_argument("--epochs", type=int, default=150)
a = ap.parse_args()

if a.mode == "collect":
    from dynamic_bicycle_env import ClutteredRacingEnv
    from mppi_planner import MPPIPlanner
    si, sn = (int(v) for v in a.shard.split("/"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(1000 + si)
    planners = {}
    S, A, V = [], [], []
    t0 = time.time()
    for ep in range(si, a.episodes, sn):
        vt = float(rng.choice([3.0, 3.5, 4.0, 4.5]))
        if vt not in planners:
            planners[vt] = MPPIPlanner(num_samples=a.K, horizon=30, v_target=vt, r_margin=0.38, device=dev)
        pl = planners[vt]; pl.reset()
        env = ClutteredRacingEnv(num_obstacles=0, track_length=1e6, track_width=4.0, seed=ep)
        env.model.p.mu = PLANT_MU
        env.reset(start_vx=float(rng.uniform(0.5, 5.0)))
        env.state = np.array([0.0, rng.uniform(-1.4, 1.4), rng.uniform(-0.6, 0.6), env.state[3],
                              rng.uniform(-0.3, 0.3), rng.uniform(-1.0, 1.0)], dtype=np.float32)
        env.actual_delta = float(rng.uniform(-0.2, 0.2))
        for k in range(a.steps):
            x = np.array(env.state, dtype=np.float32)
            u_exp = pl.plan(x, np.zeros((0, 3), dtype=np.float32))
            S.append(x); A.append(np.asarray(u_exp, dtype=np.float32)); V.append(vt)
            u_exec = np.array(u_exp, dtype=np.float32)
            if rng.random() < 0.3:     # perturbed execution -> states off the expert's own distribution
                u_exec = u_exec + np.array([rng.normal(0, 2.0), rng.normal(0, 0.15)], dtype=np.float32)
            env.step(u_exec, dt=0.02)
            if abs(env.state[1]) > 1.9 or abs(env.state[2]) > 1.2:
                break
    np.savez(os.path.join(MODELS, f"il_data_{si}.npz"), S=np.array(S), A=np.array(A), V=np.array(V))
    print(f"shard {si}: {len(S)} samples in {time.time()-t0:.0f}s", flush=True)

else:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(glob.glob(os.path.join(MODELS, "il_data_*.npz")))
    d = [np.load(f) for f in files]
    S = torch.tensor(np.concatenate([x["S"] for x in d]), device=dev)
    A = torch.tensor(np.concatenate([x["A"] for x in d]), device=dev)
    V = torch.tensor(np.concatenate([x["V"] for x in d]), device=dev, dtype=torch.float32)
    A[:, 0].clamp_(-9.5, 9.5); A[:, 1].clamp_(-0.42, 0.42)
    n = len(S); g = torch.Generator(device="cpu"); g.manual_seed(0)
    perm = torch.randperm(n, generator=g).to(dev); tr, va = perm[:int(0.9 * n)], perm[int(0.9 * n):]
    net = PiNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    sc = torch.tensor([9.5, 0.42], device=dev)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        p = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(p), 1024):
            j = p[i:i + 1024]
            loss = (((net(S[j], V[j]) - A[j]) / sc) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep % 25 == 0 or ep == 1:
            with torch.no_grad():
                vl = (((net(S[va], V[va]) - A[va]) / sc) ** 2).mean(0)
            print(f"epoch {ep:3d} train {loss.item():.4f} val_nmse ax {vl[0].item():.4f} delta {vl[1].item():.4f}", flush=True)
    with torch.no_grad():
        err = (net(S[va], V[va]) - A[va]).abs()
    meta = {"n_samples": n, "val_mae_ax": err[:, 0].mean().item(), "val_mae_delta": err[:, 1].mean().item(),
            "train_time_s": time.time() - t0}
    print(json.dumps(meta), flush=True)
    print("saved", save(net, "pi_theta.pth", meta))
