import numpy as np
import do_mpc
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
import casadi as ca
import logging
import os
import sys
import random
logging.getLogger("do-mpc").setLevel(logging.CRITICAL)
class SuppressConsoleOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# Define the system dynamics
def bicycle_dynamics(state, control):
    # Bicycle Model Parameters
    L = 2.0  # Wheelbase
    lr = 1.2  # Distance from CoM to rear axle
    v_const = control[1]  # Velocity
    delta=control[0]
    dt=0.1
    x = state[0]
    y = state[1]
    theta = state[2]
    beta = ca.arctan(lr * ca.tan(delta) / L)
    dx = v_const * ca.cos(theta)
    dy = v_const * ca.sin(theta)
    dtheta = delta#(v_const * ca.sin(beta)) / L
    return [dx, dy, dtheta]

def bicycle_dynamics_discrete(state, control):
    # Bicycle Model Parameters
    L = 2.0  # Wheelbase
    lr = 1.2  # Distance from CoM to rear axle
    v_const = control[0]  # Velocity
    delta=control[1]
    dt=0.1
    x = state[0]
    y = state[1]
    theta = state[2]
    beta = ca.arctan(lr * ca.tan(delta) / L)
    dx = v_const * ca.cos(theta)
    dy = v_const * ca.sin(theta)
    dtheta = delta
    
    x_next = x + dx * dt
    y_next = y + dy * dt
    theta_next = theta + dtheta * dt
    return [x_next, y_next, theta_next]


# Neural Network Model
class NNController(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NNController, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Define the MPC using do-mpc
def setup_mpc_model():
    model = do_mpc.model.Model('discrete')

    x = model.set_variable(var_type='_x', var_name='x')
    y = model.set_variable(var_type='_x', var_name='y')
    theta = model.set_variable(var_type='_x', var_name='theta')
    v = model.set_variable(var_type='_u', var_name='u1')
    beta = model.set_variable(var_type='_u', var_name='u2')
    model.set_rhs('x',x+v*np.cos(theta)*0.1)
    model.set_rhs('y',y+v*np.sin(theta)*0.1)
    model.set_rhs('theta',beta*0.1)
    model.setup()
    return model

def setup_mpc_controller(model, init, goal):
    model = do_mpc.model.Model('discrete')

    x = model.set_variable(var_type='_x', var_name='x')
    y = model.set_variable(var_type='_x', var_name='y')
    theta = model.set_variable(var_type='_x', var_name='theta')
    v = model.set_variable(var_type='_u', var_name='u1')
    beta = model.set_variable(var_type='_u', var_name='u2')
    model.set_rhs('x',x+v*np.cos(theta)*0.1)
    model.set_rhs('y',y+v*np.sin(theta)*0.1)
    model.set_rhs('theta',beta*0.1)
    model.setup()
    mpc = do_mpc.controller.MPC(model)

    # Setup parameters for the controller:

    setup_mpc = {'n_robust': 0,  # Robust horizon
                     'n_horizon': 10,
                     't_step': 0.1,
                     'state_discretization': 'discrete',
                     'store_full_solution': True,
                     # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
                     }

    mpc.set_param(**setup_mpc)

    # Define the objective:
    x = model.x['x']
    y = model.x['y']
    theta = model.x['theta']
    v = model.u['u1']
    beta=model.u['u2']
    # thetades=np.arctan2(-2,-1)
    # mterm = (theta-goal[0])**2 + (omega-goal[1])**2 + u**2
    # lterm = (theta-goal[0])**2 + (omega-goal[1])**2 + u**2
    mterm =20*( (x)**2 + (y)**2) + 0.005*np.arctan2(np.sin(theta),np.cos(theta))**2#ca.DM(0)#(theta)**2 + (omega)**2 + u**2
    # lterm =15*( (x)**2 + (y)**2) + 0.005*np.arctan2(np.sin(thetades-theta),np.cos(thetades-theta))**2+2*v**2+0.5*beta**2
    lterm =20*( (x)**2 + (y)**2) +2*v**2+0.5*beta**2

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u1=2, u2=0.5 )

    # # Lower bounds on states:
    # mpc.bounds['lower', '_x', 'x'] = -3
    # mpc.bounds['lower', '_x', 'y'] = -3
    # mpc.bounds['lower', '_x', 'theta'] = -30
    # # Upper bounds on states:
    # mpc.bounds['upper', '_x', 'x'] = 3
    # mpc.bounds['upper', '_x', 'y'] = 3
    # mpc.bounds['upper', '_x', 'theta'] = 30

    # # Control bounds:
    # mpc.bounds['lower', '_u', 'u1'] = -0.26*2
    # mpc.bounds['upper', '_u', 'u1'] = 0.26*2
    # mpc.bounds['lower', '_u', 'u2'] = -1.8*2
    # mpc.bounds['upper', '_u', 'u2'] = 1.8*2
    mpc.setup()

    return mpc

def train_NN_with_MPC_policies(model, mpc, n, m, epochs=700, batch_size=10):
    # Check for GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move the model to GPU if available
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    losses = []
    horizon_length = 100
    
    # Pre-compute trajectories
    all_states_trajectories = []
    all_u_mpc_trajectories = []

    logging.getLogger("do-mpc").setLevel(logging.CRITICAL)
    for init in train_initial_states:
        mpc.x0 = init
        states_trajectory = [init]
        u_mpc_trajectory = []
        for _ in range(horizon_length):
            with SuppressConsoleOutput():
                u_mpc = mpc.make_step(states_trajectory[-1])
            u_mpc_trajectory.append(u_mpc)
            next_state = np.array(bicycle_dynamics_discrete(states_trajectory[-1], u_mpc))
            states_trajectory.append(next_state)
        
        all_states_trajectories.append(states_trajectory[:-1])
        all_u_mpc_trajectories.append(u_mpc_trajectory)

    for epoch in range(epochs):
        total_loss = 0.0
        for idx in range(0, len(train_initial_states), batch_size):
            optimizer.zero_grad()
            batch_loss = 0.0
            for b in range(batch_size):
                if idx+b < len(train_initial_states):
                    states_trajectory = all_states_trajectories[idx+b]
                    u_mpc_trajectory = all_u_mpc_trajectories[idx+b]
                    states_tensor = torch.tensor(states_trajectory, dtype=torch.float32)
                    output = model(states_tensor)
                    target = torch.tensor(u_mpc_trajectory, dtype=torch.float32).squeeze(-1)
                    loss = criterion(output, target)
                    batch_loss += loss.item()
                    loss.backward()
            optimizer.step()
            total_loss += batch_loss

        avg_loss = total_loss / len(train_initial_states)
        losses.append(avg_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss}")

    return model, losses



# Visualize NN learning
def visualize_learning(mpc, model, dynamics, n, m, init, goal):
    state_mpc = init
    state_nn=init
    mpc_traj = [init]
    nn_traj = [init]
    
    for _ in range(200):
        mpc.x0 = np.array(state_mpc)
        logging.getLogger("do-mpc").setLevel(logging.CRITICAL)
        with SuppressConsoleOutput():
            u_mpc = mpc.make_step(np.array(state_mpc))
        pass
        # print(u_mpc)
        # u_mpc = mpc.make_step(np.array(state))
        # print(u_mpc)
        # state_tensor = torch.tensor(np.hstack((np.array(state).astype(np.float32), np.array(goal).astype(np.float32))), dtype=torch.float32)
        state_tensor = torch.tensor(np.array(state_nn).astype(np.float32))

        u_nn = model(state_tensor).detach().numpy()
        # next_state_mpc = np.array(state) + 0.1 * np.array(dynamics(state, u_mpc))
        # print(next_state_mpc)
        # next_state_nn = np.array(state) + 0.1 * np.array(dynamics(state, u_nn))
        next_state_mpc = np.array(bicycle_dynamics_discrete(state_mpc, u_mpc)).reshape(n)#np.array(state) + 0.1 * dynamics(np.array(state), u_mpc)
        # print(next_state_mpc)
        next_state_nn = np.array(bicycle_dynamics_discrete(state_nn, u_nn)).reshape(n)
        # next_state_nn = np.array(state) + 0.1 * dynamics(np.array(state), u_nn)
        mpc_traj.append(next_state_mpc)
        nn_traj.append(next_state_nn)
        state_mpc = next_state_mpc
        state_nn=next_state_nn

    mpc_traj = np.array(mpc_traj)
    nn_traj = np.array(nn_traj)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(mpc_traj[:, 0], label='MPC')
    ax[0].plot(nn_traj[:, 0], label='NN')
    ax[0].legend()
    ax[0].set_title('X')

    ax[1].plot(mpc_traj[:, 1], label='MPC')
    ax[1].plot(nn_traj[:, 1], label='NN')
    ax[1].legend()
    ax[1].set_title('Y')

    plt.tight_layout()
    plt.show()

# Main
n = 3
m = 2
init = [round(random.random(), 2) for _ in range(3)]#[1.0, 2.0, 1]
goal = [0.0, 0.0, 0]

# Training data (randomly generated)
num_samples = 2000
train_initial_states = [np.random.uniform(-np.pi/3, np.pi/3, n) for _ in range(num_samples)]
print(train_initial_states)
train_goal_states = [np.random.uniform(0, 0, n) for _ in range(num_samples)]
# train_initial_states=[np.array(init)]
# print(train_initial_states)
model_do_mpc = setup_mpc_model()
mpc = setup_mpc_controller(model_do_mpc, init, goal)
nn_model = NNController(n, m)
nn_model, losses = train_NN_with_MPC_policies(nn_model, mpc, n, m,2000)
visualize_learning(mpc, nn_model, bicycle_dynamics, n, m, init, goal)
torch.save(nn_model,"model/IL_model.pth")
print(train_initial_states)