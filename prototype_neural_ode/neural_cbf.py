import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import qpsolvers
import imageio

# Function to check if a point lies in any of the circles
def in_any_circle(point, centers, radii):
    for center, radius in zip(centers, radii):
        if torch.sum((point - center)**2) < radius**2:
            return True
    return False

# 1. Generate Training Data
num_samples = 10000
data = 1*(torch.rand(num_samples, 2) * 2 - 1)  # samples in [-1, 1] range

centers = [torch.tensor([0.2, -0.5]), torch.tensor([0.5, -0.2])] #[torch.tensor([0.5, 0.5])]  # Example centers
radii = [0.2, 0.2]  # Example radii

centers = [torch.tensor([0.0, 0.0])] #[torch.tensor([0.5, 0.5])]  # Example centers
radii = [0.5]  # Example radii

# # Given N random circles
# N = 10  # Number of random circles
# centers = torch.rand(N, 2) * 2 - 1  # Random centers in [-1, 1] range
# radii = torch.rand(N) * 0.5  # Random radii between 0 and 0.5

num_samples = 10000
data = torch.rand(num_samples, 2) * 2 - 1  # samples in [-1, 1] range

labels = torch.tensor([-1.0 if in_any_circle(point, centers, radii) else 1.0 for point in data])

# 2. Define the Neural Network Architecture
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

def plot_heatmap(epoch, predictions, resolution=100):
    fig, ax = plt.subplots()
    ax.imshow(predictions, cmap='RdYlGn', extent=(-1, 1, -1, 1))
    ax.set_title(f'Epoch {epoch}')
    plt.savefig(f'images/heatmap_epoch_{epoch}.png')
    plt.close(fig)


model = SimpleNN()

# 3. Train the Neural Network
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 150
resolution = 500
x = np.linspace(-1, 1, resolution)
y = np.linspace(-1, 1, resolution)
X, Y = np.meshgrid(x, y)
test_data = torch.FloatTensor(np.c_[X.ravel(), Y.ravel()])
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(data).squeeze()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    predictions = model(test_data).detach().numpy().reshape(resolution, resolution)
    plot_heatmap(epoch, predictions)
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Create a video from the saved heatmaps
filenames = [f'images/heatmap_epoch_{i}.png' for i in range(epochs)]
with imageio.get_writer('videos/ICBF_evolution.mp4', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)


# 4. Plot a Heatmap
resolution = 500
x = np.linspace(-1, 1, resolution)
y = np.linspace(-1, 1, resolution)
X, Y = np.meshgrid(x, y)
test_data = torch.FloatTensor(np.c_[X.ravel(), Y.ravel()])
torch.save(model,"model/ICBF_model.pth")
predictions = model(test_data).detach().numpy().reshape(resolution, resolution)
# plt.imshow(predictions, cmap='RdYlGn', extent=(-1, 1, -1, 1))
# plt.colorbar()
# plt.title('Heatmap')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()

# # Placeholder for the dynamical system function f
# # Dynamics of the system
# def f(z, u=None):
#     if u is None:
#         u = torch.tensor(0.0)
#     return torch.tensor([z[1], -z[0] + u])


# # Compute the Lie derivative
# def compute_Lie_derivative(model, z, u):
#     z.requires_grad_(True)  # Make sure z has gradients
#     h = model(z)
#     h.backward()  # Compute the gradient of h w.r.t z
#     grad_h = z.grad
#     return torch.dot(grad_h, f(z, u))

# # Test the Lie derivative computation
# z = torch.tensor([1.0, 2.0])
# u = torch.tensor([0.5])
# Lie_derivative = compute_Lie_derivative(model, z, u)
# print(Lie_derivative)

# def compute_Lie_derivatives(model, z):
#     z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True)  # Convert z to a PyTorch tensor
#     h = model(z_tensor)
#     h.backward()
#     Lf_h = torch.dot(z_tensor.grad, f(z_tensor, None))
#     Lg_h = z_tensor.grad
#     return Lf_h, Lg_h

# # Define the QP parameters
# Q = np.array([[1.0, 0.0], [0.0, 1.0]])
# c = np.array([0.0, 0.0])
# A = np.array([[-1.0, 0.0]])  # Will be replaced with the actual constraint from Lie derivatives
# b = np.array([0.0])  # Will be replaced with the actual constraint value

# # Solve the QP
# def solve_QP(model, z):
#     Lf_h, Lg_h = compute_Lie_derivatives(model, z)
#     alpha_h = 1.0*model(torch.tensor(z, dtype=torch.float32, requires_grad=True))  # Some class-K function of h(z), for simplicity just a negative constant here
#     A[0] = -Lg_h.detach().numpy()
#     b[0] = -(Lf_h.detach().numpy() + alpha_h.detach().numpy())
#     u_star = qpsolvers.solve_qp(Q, c, A, b, solver='quadprog')
#     return u_star  

# z = np.array([1.0, 2.0])
# safe_control_input = solve_QP(model, z)
# print(safe_control_input)

#For h(x)
# plt.imshow(predictions, cmap='RdYlGn', extent=(0, 1, 0, 1))
# plt.colorbar()
# plt.title(r'Projection of $h(x,u)$ on $x$',fontsize=15)
# plt.xlabel('$X$',fontsize=15)
# plt.ylabel('$Y$',fontsize=15)
# plt.tick_params(axis='both', which='major', labelsize=15)

# plt.show()

# #For h(u)
plt.imshow(predictions, cmap='RdYlGn', extent=(-1, 1, -1, 1))
plt.colorbar()
plt.title(r'Projection of $h(x,u)$ on $u$',fontsize=15)
plt.xlabel('$u_1$',fontsize=15)
plt.ylabel('$u_2$',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()