import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import qpsolvers
import sys
import __main__

# Define the network architectures so they can be unpickled properly
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

# Inject into __main__ so torch.load can find the class definitions
__main__.SimpleNN = SimpleNN
__main__.BicycleODEFunc = BicycleODEFunc
__main__.NNController = NNController

# Load the models with weights_only=False to allow unpickling the custom classes
h = torch.load('model/ICBF_model_learned.pth', weights_only=False)
h.eval()

Fs = torch.load('model/system_model_neural_ode_learned.pth', weights_only=False)
Fs.eval()

u_N = torch.load('model/IL_model_learned.pth', weights_only=False)
u_N.eval()


def bicycle_f(z_tensor):
    # z_tensor shape: (1, 3) — (x, y, theta)
    theta = z_tensor[0, 2]
    # f0(x) = 0, f1(x) = [[cos(theta), 0], [sin(theta), 0], [0, 1]]
    # Returns f1(x) as (3, 2) tensor — for control-affine: x_dot = f1(x) @ u
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    f1 = torch.stack([
        torch.stack([cos_t, torch.zeros_like(cos_t)]),
        torch.stack([sin_t, torch.zeros_like(sin_t)]),
        torch.stack([torch.zeros_like(cos_t), torch.ones_like(cos_t)])
    ])  # shape (3, 2)
    return f1


def compute_phi(z_tensor, u_tensor, alpha_phi=0.5):
    # z_tensor: (1,3) with requires_grad=True
    # u_tensor: (1,2) with requires_grad=True
    u_N_out = u_N(z_tensor)  # pi_theta(x), shape (1,2)
    # term1: (d pi_theta / dx) * f(x,u)
    # Compute Jacobian of u_N_out w.r.t. z_tensor: shape (2,3)
    jac_rows = []
    for i in range(2):
        g = torch.autograd.grad(u_N_out[0, i], z_tensor, retain_graph=True, create_graph=True)[0]
        jac_rows.append(g)  # (1,3)
    jac = torch.cat(jac_rows, dim=0)  # (2,3)
    f1 = bicycle_f(z_tensor)  # (3,2)
    f_val = f1 @ u_tensor.T  # (3,1) — x_dot = f1 @ u
    term1 = jac @ f_val  # (2,1)
    # term2: (alpha_phi/2) * (pi_theta(x) - u)
    term2 = (alpha_phi / 2.0) * (u_N_out - u_tensor)  # (1,2)
    phi_val = term1.T + term2  # (1,2)
    return phi_val, u_N_out


def input_barrier(u_tensor, phi_val, u_max=1.0, gamma_u=0.1):
    hu = u_max**2 - (u_tensor * u_tensor).sum()
    pu = -2.0 * u_tensor  # (1,2)
    du = (2.0 * u_tensor * phi_val).sum() - gamma_u * hu
    return hu, pu, du


def compute_he_constraints(z_tensor, u_tensor, phi_val, alpha=0.1, gamma_e=0.1):
    # hx(x): barrier network, 2D position input only
    h_out = h(z_tensor[:, :2])  # scalar, (1,1)
    # px(x) = d(hx)/dx * f1(x): shape (1,2)
    h_grads = torch.autograd.grad(h_out, z_tensor, create_graph=True)[0]  # (1,3)
    f1 = bicycle_f(z_tensor)  # (3,2)
    px = h_grads[:, :3] @ f1  # (1,2) — this is px(x)^T

    # h_dot_x = px^T @ u (control-affine: h_dot_x = px(x) . u)
    h1 = (px * u_tensor).sum()  # scalar — this is h_dot_x(x,u)

    # he(x,u) = h_dot_x + alpha * h_out.squeeze()
    he = h1 + alpha * h_out.squeeze()

    # pe(x,u) = d(he)/du = px(x) (by Lemma 3: pe = px, independent of u)
    pe = px  # (1,2)

    # de from eq (17) applied to he:
    # de = -(d(he)/dx * f(x,u) + d(he)/du * phi + gamma_e * he)
    # d(he)/du = pe = px (already have it)
    # d(he)/dx * f(x,u): differentiate h1 + alpha*h_out w.r.t. x, then dot with f(x,u)
    he_for_grad = h1 + alpha * h_out.squeeze()
    dhe_dx = torch.autograd.grad(he_for_grad, z_tensor, create_graph=True)[0]  # (1,3)
    f_val = f1 @ u_tensor.T  # (3,1)
    lie_he = (dhe_dx[:, :3] @ f_val).squeeze()  # scalar

    phi_contrib = (pe * phi_val).sum()  # scalar
    de = -(lie_he + phi_contrib + gamma_e * he)

    return pe, de, h_out, h1, he


def solve_v_star(pe, de, pu, du):
    # pe: (1,2), de: scalar
    # pu: (1,2), du: scalar
    P = np.eye(2, dtype=np.float64)
    q_vec = np.zeros(2, dtype=np.float64)
    pe_np = pe.detach().numpy().flatten().astype(np.float64)
    pu_np = pu.detach().numpy().flatten().astype(np.float64)
    de_val = float(de.item())
    du_val = float(du.item())
    G = -np.vstack([pe_np, pu_np])  # (2,2)
    h_vec = -np.array([de_val, du_val], dtype=np.float64)
    try:
        v = qpsolvers.solve_qp(P, q_vec, G, h_vec, solver='quadprog')
        if v is None:
            print(f"WARNING: QP returned None (de={de_val:.4f}, du={du_val:.4f}). Defaulting to zero.")
            v = np.zeros(2)
    except Exception as e:
        print(f"WARNING: QP failed ({e}). Defaulting to zero.")
        v = np.zeros(2)
    return torch.tensor(v, dtype=torch.float32).unsqueeze(0)  # (1,2)


def compute_u_safe(z, u, alpha=0.1, alpha_phi=0.5, gamma_e=0.1, gamma_u=0.1, u_max=1.0):
    z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
    u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(0).requires_grad_(True)

    phi_val, u_N_out = compute_phi(z_tensor, u_tensor, alpha_phi)
    pe, de, h_out, h1, he = compute_he_constraints(z_tensor, u_tensor, phi_val, alpha, gamma_e)
    hu, pu, du = input_barrier(u_tensor, phi_val, u_max, gamma_u)

    v_star = solve_v_star(pe, de, pu, du)

    # u_dot = phi + v* (eq. 22 of paper)
    u_dot = phi_val + v_star  # (1,2)

    return (u_dot.detach().numpy().flatten(),
            phi_val.detach().numpy().flatten(),
            v_star.detach().numpy().flatten(),
            float(h_out.item()),
            float(hu.item()),
            float(he.item()))
