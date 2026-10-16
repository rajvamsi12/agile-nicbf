import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# import qpsolvers
import do_mpc
import imageio
from torchdiffeq import odeint
import random
import logging
import os
import sys
import casadi as ca
import pandas as pd
import time
from tqdm import tqdm

os.makedirs("outputs", exist_ok=True)
os.makedirs("loss", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 2. Define the ICBF Neural Network Architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x
    
# Function to check if a point lies in any of the circles
def in_any_circle(point, centers, radii):
    for center, radius in zip(centers, radii):
        if torch.sum((point - center)**2) < radius**2:
            return True
    return False


# Constants for the bicycle model
L = 2.0  # wheelbase
lr = 1.2  # distance from CoG to rear axle

# 1. Define the bicycle model dynamics
def bicycle_dynamics(t, state, control):
    x, y, theta = state[..., 0], state[..., 1], state[..., 2]
    
    t_idx = (t / 0.1).long()  # Assuming a time step of 0.1
    v, delta = control[t_idx][0], control[t_idx][1]
    
    # beta = torch.atan(lr * torch.tan(delta) / L)
    dx = v * torch.cos(theta).unsqueeze(-1)
    dy = v * torch.sin(theta).unsqueeze(-1)
    dtheta =  delta.unsqueeze(-1)
    # dv = a.unsqueeze(-1)
    
    # Ensure all tensors have two dimensions
    dx = dx.unsqueeze(-1) if dx.dim() == 1 else dx
    dy = dy.unsqueeze(-1) if dy.dim() == 1 else dy
    dtheta = dtheta.unsqueeze(-1) if dtheta.dim() == 1 else dtheta
    # dv = dv.unsqueeze(-1) if dv.dim() == 1 else dv
    
    return torch.cat([dx, dy, dtheta], dim=-1)


# 2. Implement the Neural ODE
class BicycleODEFunc(nn.Module):
    def __init__(self):
        super(BicycleODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, t, state_control):
        return self.net(state_control)
    
## For imitation init
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

def bicycle_dynamics_torch(state, control, dt=0.1):
    # state: (B, 3) — x, y, theta
    # control: (B, 2) — v, delta
    x, y, theta = state[:, 0], state[:, 1], state[:, 2]
    v, delta = control[:, 0], control[:, 1]
    x_next = x + v * torch.cos(theta) * dt
    y_next = y + v * torch.sin(theta) * dt
    theta_next = theta + delta * dt
    theta_next = (theta_next + np.pi) % (2 * np.pi) - np.pi
    return torch.stack([x_next, y_next, theta_next], dim=1)


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
    setup_mpc = {'n_robust': 0,  # Robust horizon
                     'n_horizon': 10,
                     't_step': 0.1,
                     'state_discretization': 'discrete',
                     'store_full_solution': True,
                     # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
                     }

    mpc.set_param(**setup_mpc)
    x = model.x['x']
    y = model.x['y']
    theta = model.x['theta']
    v = model.u['u1']
    beta=model.u['u2']
    mterm =20*( (x)**2 + (y)**2) + 0.005*np.arctan2(np.sin(theta),np.cos(theta))**2
    lterm =20*( (x)**2 + (y)**2) +2*v**2+0.5*beta**2
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u1=2, u2=0.5 )
    mpc.setup()
    return mpc








########################################################### Parameters ######################################

# Parameters for ICBF
num_samples_icbf = 10000
data = 1*(torch.rand(num_samples_icbf, 2) * 2 - 1)  # samples in [-1, 1] range
centers = [torch.tensor([0.5, -0.5])] #[torch.tensor([0.5, 0.5])]  # Example centers
radii = [0.2]  # Example radii
# labels will be generated dynamically after imitation learning
epochs_icbf = int(os.environ.get("EPOCHS_ICBF", 5000))


#Parameters for system learning
# Tune it in function train_neural_ode ( num_time_steps_system, num_samples_system, epochs_system)

#Parameters for Imitation Learning
n = 3
m = 2
init = [round(random.random(), 2) for _ in range(3)]#[1.0, 2.0, 1]
goal = [0.0, 0.0, 0]
num_samples_imitation = 2000
train_initial_states = [np.random.uniform(-np.pi/3, np.pi/3, n) for _ in range(num_samples_imitation)]
train_goal_states = [np.random.uniform(0, 0, n) for _ in range(num_samples_imitation)]
model_do_mpc = setup_mpc_model()
mpc = setup_mpc_controller(model_do_mpc, init, goal)
try:
    nn_model_imitation = torch.load("model/IL_model_learned.pth", map_location=device)
except Exception:
    nn_model_imitation = NNController(n, m)
epochs_imitation = int(os.environ.get("EPOCHS_IMITATION", 1000))











######################################################Training #######################################################

#####Training for system learning using neural ode##########
def train_neural_ode():
    losses=[]
    try:
        model = torch.load("model/system_model_neural_ode_learned.pth", map_location=device)
    except Exception:
        model = BicycleODEFunc()
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    num_time_steps_system = 100
    t = torch.linspace(0, 10, num_time_steps_system)
    control_dim = 2
    num_samples_system=150
    inits=torch.rand((num_samples_system, 3))
    epochs_system = int(os.environ.get("EPOCHS_SYSTEM", 80000))
    for epoch in tqdm(range(epochs_system), desc="Training Neural ODE (F_theta)", unit="epoch"):
        loss=0.0
        for i in range(num_samples_system):
            init_state=inits[i].unsqueeze(0)
            control = torch.rand((num_time_steps_system, control_dim))
            pred_trajectory = odeint(lambda t, y: model(t, torch.cat([y, control[(t / 0.1).long()].unsqueeze(0)], dim=1)), init_state, t, method='euler', options={'step_size': 0.1})
            true_trajectory = odeint(lambda t, y: bicycle_dynamics(t, y, control), init_state, t, method='euler', options={'step_size': 0.1})
            loss+=torch.mean((true_trajectory - pred_trajectory) ** 2)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            tqdm.write(f"[Neural ODE] Epoch {epoch}/{epochs_system} | Loss: {loss.item():.6f}")
    return model, inits, control, losses
model_system, inits, control, losses_system = train_neural_ode()
losses_np = [loss.item() for loss in losses_system]
losses_df = pd.DataFrame(losses_np, columns=['Loss'])
losses_df.to_csv('loss/losses_system.csv', index=False)
torch.save(model_system,"model/system_model_neural_ode.pth")



#######Training for imitation learning#############
def train_NN_with_MPC_policies(model, mpc, n, m, epochs=700, batch_size=10):
    # Check for GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # Move the model to GPU if available
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    losses = []
    horizon_length = 100
    
    all_states_trajectories = []
    all_u_mpc_trajectories = []

    logging.getLogger("do-mpc").setLevel(logging.CRITICAL)
    for init in tqdm(train_initial_states, desc="Generating MPC trajectories", unit="traj"):
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

    for epoch in tqdm(range(epochs), desc="Training Imitation Controller (pi_theta)", unit="epoch"):
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
            tqdm.write(f"[Imitation] Epoch {epoch+1}/{epochs} | Avg imitation Loss: {avg_loss:.6f}")

    return model, losses
nn_model_imitation, losses_imitation = train_NN_with_MPC_policies(nn_model_imitation, mpc, n, m,epochs_imitation)
# losses_np = [loss.item() for loss in losses_imitation]
losses_df = pd.DataFrame(losses_imitation, columns=['Loss'])
losses_df.to_csv('loss/losses_imitation.csv', index=False)
torch.save(nn_model_imitation,"model/IL_model.pth")

# --- Retrain ICBF on forward-invariant labels ---
# Training for ICBF
from forward_invariant_labels import generate_labels, plot_label_comparison
nn_model_imitation.eval()
t0 = time.time()
labels, old_labels, flipped_to_unsafe, flipped_to_safe = generate_labels(
    points=data,
    pi_theta=nn_model_imitation,
    centers=centers,
    radii=radii,
    T_label=20,
    dt=0.1,
    n_headings=8,
    u_max=4.0,   # match u_max used in main.py
    dynamics_fn=bicycle_dynamics_torch,
    in_any_circle_fn=in_any_circle,
)
print(f"Labeling time: {time.time() - t0:.1f}s")
nn_model_imitation.train()

plot_label_comparison(data, old_labels, labels, centers, radii,
                        save_path="outputs/label_comparison.png")
tqdm.write(f"Points relabeled safe->unsafe (forward-invariance violations caught): "
           f"{len(flipped_to_unsafe)} / {len(data)} "
           f"({100*len(flipped_to_unsafe)/len(data):.2f}%)")
tqdm.write(f"Points relabeled unsafe->safe (sanity check, should be ~0): "
           f"{len(flipped_to_safe)} / {len(data)}")

# Save these counts to a small CSV for the paper
relabel_stats_df = pd.DataFrame({
    "metric": ["total_points", "flipped_safe_to_unsafe", "flipped_unsafe_to_safe"],
    "value": [len(data), len(flipped_to_unsafe), len(flipped_to_safe)]
})
os.makedirs("loss", exist_ok=True)
relabel_stats_df.to_csv("loss/relabeling_stats.csv", index=False)

# Train ICBF
model = SimpleNN()
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Ensure data and labels are on the correct device
data = data.to(device)
labels = labels.to(device)

resolution = 500
x = np.linspace(-1, 1, resolution)
y = np.linspace(-1, 1, resolution)
X, Y = np.meshgrid(x, y)
losses_icbf=[]
test_data = torch.FloatTensor(np.c_[X.ravel(), Y.ravel()]).to(device)

for epoch in tqdm(range(epochs_icbf), desc="Training ICBF Barrier (h_theta)", unit="epoch"):
    optimizer.zero_grad()
    outputs = model(data).squeeze()
    loss = criterion(outputs, labels)
    losses_icbf.append(loss)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        tqdm.write(f"[ICBF] Epoch {epoch+1}/{epochs_icbf} | Loss: {loss.item():.6f}")

torch.save(model, "model/ICBF_model_learned_v2_forward_invariant.pth")
losses_np = [loss.item() for loss in losses_icbf]
losses_df = pd.DataFrame(losses_np, columns=['Loss'])
losses_df.to_csv('loss/losses_icbf.csv', index=False)

# STEP 3 - Save the heatmap figure
predictions = model(test_data).cpu().detach().numpy().reshape(resolution, resolution)
plt.figure(figsize=(8, 8))
plt.imshow(predictions, cmap='RdYlGn', extent=(-1, 1, -1, 1))
plt.colorbar()
plt.title('Learned ICBF Heatmap (Forward-Invariant Labeling)', fontsize=14)
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/h_theta_heatmap_forward_invariant.png", dpi=300, bbox_inches='tight')
plt.close()
