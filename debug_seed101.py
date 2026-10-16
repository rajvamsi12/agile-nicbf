import sys, os
sys.path.insert(0, '/home/raj/agile_nicbf')
import numpy as np
import torch
from dynamic_bicycle_env import ClutteredRacingEnv
from mppi_planner import MPPIPlanner
from test_fix import RobustSafeController

env = ClutteredRacingEnv(num_obstacles=10, track_length=50.0, track_width=4.0, seed=101)
env.model.p.mu = 1.0
state = env.reset(start_vx=3.0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
planner = MPPIPlanner(num_samples=50, horizon=30, v_target=6.0, r_margin=0.26, device=device)
shield = RobustSafeController(model_path='/home/raj/agile_nicbf/models/neural_icbf_6d.pth', mode='neural_icbf', r_margin=0.26, alpha1=2.5, alpha2=3.5, alpha_w=2.5, device=device)

current_u = np.array([env.actual_ax, env.actual_delta], dtype=np.float32)

for step in range(1, 100):
    perceived = env.get_perceived_obstacles()
    obs_state = env.get_noisy_state()
    u_nom = planner.plan(obs_state, perceived, barrier_net=shield.barrier_net)
    u_exec, qi = shield.filter_control(obs_state, current_u, u_nom, perceived, dt=0.02)
    state, info = env.step(u_exec, dt=0.02)
    current_u = info['actual_u']
    
    if step >= 80:
        print('step', step, 'x=', round(state[0], 2), 'y=', round(state[1], 2), 'vx=', round(state[3], 2),
              'u_nom=', [round(x, 2) for x in u_nom], 'u_exec=', [round(x, 2) for x in u_exec],
              'status=', qi['qp_status'], 'min_h=', round(qi['min_h_val'], 3),
              'slk=', round(qi['slack_max'], 2), 'collision=', info['collision'])
    if info['collision']:
        print('COLLISION! info=', info)
        print('perceived obstacles=', perceived)
        break
