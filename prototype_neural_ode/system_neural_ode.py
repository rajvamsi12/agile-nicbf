import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Constants for the bicycle model
L = 2.0  # wheelbase
lr = 1.2  # distance from CoG to rear axle

# 1. Define the bicycle model dynamics
def bicycle_dynamics(t, state, control):
    x, y, theta = state[..., 0], state[..., 1], state[..., 2]
    
    t_idx = (t / 0.1).long()  # Assuming a time step of 0.1
    v, delta = control[t_idx][0], control[t_idx][1]
    
    beta = torch.atan(lr * torch.tan(delta) / L)
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
    
# def sample_initial_conditions_and_controls(num_samples, control_dim, num_time_steps):
#     # Sample random initial conditions
#     initial_conditions = torch.rand((num_samples, 4))  # Assuming state dimension is 4
#     # Sample random control sequences
#     controls = torch.rand((num_samples, num_time_steps, control_dim))
#     return initial_conditions, controls

# def train_neural_ode(t,epochs=1000, num_samples=10):
#     model=BicycleODEFunc()
#     t = torch.linspace(0, 10, 100)  # Define the time tensor here
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     for epoch in range(epochs):
#         # Sample random initial conditions and control sequences
#         initial_conditions, controls = sample_initial_conditions_and_controls(num_samples, control_dim=2, num_time_steps=100)
#         total_loss = 0
#         for i in range(num_samples):
#             true_trajectory = odeint(lambda t, y: bicycle_dynamics(t, y, controls[i][(t / 0.1).long()].unsqueeze(0)), initial_conditions[i], t, method='euler', options={'step_size': 0.1})
#             pred_trajectory = odeint(lambda t, y: model(t, torch.cat([y, controls[i][(t / 0.1).long()].unsqueeze(0)], dim=1)), initial_conditions[i], t, method='euler', options={'step_size': 0.1})
#             loss = torch.mean((true_trajectory - pred_trajectory) ** 2)
#             total_loss += loss
#         total_loss /= num_samples
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
#         if epoch % 100 == 0:
#             print(f"Epoch {epoch}, Loss: {total_loss.item()}")
#     return model
# t = torch.linspace(0, 10, 100)
# model = train_neural_ode(t)


# 3. Train the Neural ODE
def train_neural_ode():
    model = BicycleODEFunc()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    num_time_steps = 100
    t = torch.linspace(0, 10, num_time_steps)
    control_dim = 2
    # control = torch.rand((num_time_steps, control_dim))
    # t = torch.linspace(0, 10, 100)
    init_state = torch.tensor([[0.0, 0.0, 1.0]])
    # control = torch.tensor([[0.1, 0.2] for _ in range(100)])  # example control inputs
    num_samples=200
    inits=torch.rand((num_samples, 3))

    for epoch in range(10000):
        loss=0.0
        for i in range(num_samples):
            init_state=inits[i].unsqueeze(0)
            control = torch.rand((num_time_steps, control_dim))

            # print(init_state)
            pred_trajectory = odeint(lambda t, y: model(t, torch.cat([y, control[(t / 0.1).long()].unsqueeze(0)], dim=1)), init_state, t, method='euler', options={'step_size': 0.1})
            true_trajectory = odeint(lambda t, y: bicycle_dynamics(t, y, control), init_state, t, method='euler', options={'step_size': 0.1})
            loss+=torch.mean((true_trajectory - pred_trajectory) ** 2)
        # loss = loss+criterion(pred_trajectory, true_trajectory)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model, inits, control

# model, inits, control = train_neural_ode()
model = torch.load('model/system_model_neural_ode_learned.pth')
model.eval()  # Set the model to evaluation mode

# 4. Plot the results
num_time_steps = 100
t = torch.linspace(0, 10, num_time_steps)
control_dim = 2
plt.figure(figsize=(10, 6))

for _ in range(20):
    init_states=0.5*(2*torch.rand((1, 3))+1)
    control = 1*torch.rand((num_time_steps, control_dim))

    init_state=init_states[0].unsqueeze(0)
    true_trajectory = odeint(lambda t, y: bicycle_dynamics(t, y, control), init_state, t, method='euler', options={'step_size': 0.1})
    pred_trajectory = odeint(lambda t, y: model(t, torch.cat([y, control[(t / 0.1).long()].unsqueeze(0)], dim=1)), init_state, t, method='euler', options={'step_size': 0.1})


    # plt.figure(figsize=(10, 6))
    if _==0:
        plt.plot(true_trajectory[:, 0, 0].detach().numpy(), true_trajectory[:, 0, 1].detach().numpy(),color='red', label='True Dynamics')
        plt.plot(pred_trajectory[:, 0, 0].detach().numpy(), pred_trajectory[:, 0, 1].detach().numpy(),color='blue', label='Neural ODE')
    else:
        plt.plot(true_trajectory[:, 0, 0].detach().numpy(), true_trajectory[:, 0, 1].detach().numpy(),color='red')
        plt.plot(pred_trajectory[:, 0, 0].detach().numpy(), pred_trajectory[:, 0, 1].detach().numpy(),color='blue')  
    # plt.plot(true_trajectory[:, 0, 0].detach().numpy(), true_trajectory[:, 0, 1].detach().numpy(), label='True Dynamics')
    # plt.plot(pred_trajectory[:, 0, 0].detach().numpy(), pred_trajectory[:, 0, 1].detach().numpy(), '--', label='Neural ODE')
plt.legend()
plt.xlabel('$X$',fontsize=15)
plt.ylabel('$Y$',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.title('Bicycle Model vs Neural ODE',fontsize=15)
plt.show()

# torch.save(model,"model/system_model_neural_ode.pth")