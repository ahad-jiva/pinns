import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('mps' if torch.mps.is_available() else 'cpu')

# grf covariance matrix generator 
def grf_kernel(points, l=0.2, domain_size=1.0, var=1):

    #x = np.linspace(0, size[0], steps)
    #y = np.linspace(0, size[1], steps)
    #X, Y = np.meshgrid(x, y)
    #points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    
    #dist_matrix = cdist(points, points, metric='euclidean')
    x = np.linspace(0, domain_size, points)
    dist_matrix = np.abs(x[:, None] - x[None, :])
    kernel_matrix = var * np.exp(-dist_matrix ** 2 / (2 * l ** 2))
    kernel_matrix += 1e-6 * np.eye(points)
    L = np.linalg.cholesky(kernel_matrix)
    return x, L

def grf_field(cov_matrix, batch_size):
    xi = np.random.randn(batch_size, cov_matrix.shape[0])
    Z = 0 + xi @ L.T
    return Z

def antidiff_trapz(a_batch, x):
    dx = x[1] - x[0]
    u = np.zeros_like(a_batch)
    u[:, 1:] = np.cumsum((a_batch[:, :-1] + a_batch[:, 1:]) / 2 * dx, axis=1)
    return u

class DeepONet(nn.Module):
    def __init__(self, m=100, p=64):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(m, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, p)
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, p)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u, y):
        b = self.branch(u)
        t = self.trunk(y)
        out = torch.einsum('bp,qp->bq', b, t)
        return out + self.bias

sensor_points = 100
batch_size = 64
domain_length = 5.0
epochs = 5000

x_sensors, L = grf_kernel(sensor_points, l=0.2, domain_size = domain_length)

y_query = torch.tensor(x_sensors, dtype=torch.float32).unsqueeze(-1).to(device)

model = DeepONet(m=100, p=64).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in tqdm(range(epochs)):
    a_batch = grf_field(L, batch_size)
    u_true = antidiff_trapz(a_batch, x_sensors)

    u_in = torch.tensor(a_batch, dtype=torch.float32, device=device)
    u_target = torch.tensor(u_true, dtype=torch.float32, device=device)

    pred = model(u_in, y_query)
    loss = loss_fn(pred, u_target)

    opt.zero_grad()
    loss.backward()
    opt.step()

# testing on a simple function
model.eval()

#a_test = np.where(x_sensors < 2.5, -1.0, 1.0)
a_test = np.sin(40 * np.pi * x_sensors)
a_test_batch = a_test[None, :]

#u_true = np.abs(x_sensors - 2.5) - 2.5
u_true = (-1/(40 * np.pi)) * np.cos(40 * np.pi * x_sensors) + (1/(40 * np.pi))

y_query_test = torch.tensor(x_sensors, dtype=torch.float32).unsqueeze(-1).to(device)

x_finer = np.linspace(0, domain_length, 500)
y_query_fine = torch.tensor(x_finer, dtype=torch.float32).unsqueeze(-1).to(device)
#u_true_fine = np.abs(x_finer - 2.5) - 2.5
u_true_fine = (-1/(40 * np.pi)) * np.cos(40 * np.pi * x_finer) + (1/(40 * np.pi))


with torch.no_grad():
    u_in = torch.tensor(a_test_batch, dtype=torch.float32, device=device)

    pred_same_grid = model(u_in, y_query_test).cpu().numpy().squeeze(0)
    pred_finer_grid = model(u_in, y_query_fine).cpu().numpy().squeeze(0)

mse_same = np.mean((pred_same_grid - u_true) ** 2)
mse_finer = np.mean((pred_finer_grid - u_true_fine) ** 2)

print(f"mse on same grid = {mse_same:.6f}")
print(f"mse on finer grid = {mse_finer:.6f}")

plt.plot(x_sensors, pred_same_grid)
plt.plot(x_finer, pred_finer_grid)
plt.plot(x_finer, u_true_fine)
plt.show()
