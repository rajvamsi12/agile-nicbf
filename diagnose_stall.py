import sys, os
sys.path.insert(0, '/home/raj/agile_nicbf')
import numpy as np
import torch
from dynamic_bicycle_env import ClutteredRacingEnv
from mppi_planner import MPPIPlanner
from safe_controller import NeuralICBFController

seed = 138
torch.manual_seed(seed)
np.random.seed(seed)

env = ClutteredRacingEnv(num_obstacles=10, track_length=50.0, track_width=4.0, seed=seed)
env.model.p.mu = 1.0
state = env.reset(start_vx=3.0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
planner = MPPIPlanner(num_samples=50, horizon=30, v_target=6.0, r_margin=0.26, device=device)
shield = NeuralICBFController(model_path='/home/raj/agile_nicbf/models/neural_icbf_6d.pth', mode='neural_icbf', r_margin=0.26, alpha1=2.5, alpha2=3.5, alpha_w=2.5, device=device)

current_u = np.array([env.actual_ax, env.actual_delta], dtype=np.float32)

for step in range(1, 350):
    perceived = env.get_perceived_obstacles()
    obs_state = env.get_noisy_state()
    u_nom = planner.plan(obs_state, perceived, barrier_net=shield.barrier_net)
    u_exec, qi = shield.filter_control(obs_state, current_u, u_nom, perceived, dt=0.02)
    state, info = env.step(u_exec, dt=0.02)
    current_u = info['actual_u']
    
    if step % 20 == 0 or state[3] < 1.0 or qi['wall_fallback']:
        print(f'{step:03d} | x={state[0]:.2f}, y={state[1]:.2f}, psi={state[2]:.2f}, vx={state[3]:.2f} | nom=[{u_nom[0]:.2f}, {u_nom[1]:.2f}] | safe=[{u_exec[0]:.2f}, {u_exec[1]:.2f}] | st={qi["qp_status"]} fb={qi["wall_fallback"]} clr={qi["wall_clearance_m"]:.2f} w_mg={qi["wall_margin_min"]:.2f} slk={qi["slack_max"]:.2f} n_obs={qi["n_obstacle_rows"]}')
    if info['collision'] or info['out_of_bounds'] or info['success']:
        print('Terminal:', info)
        break
