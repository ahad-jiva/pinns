import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)


# using dirichlet boundary conditions to avoid full fourier series solution

# ground truth function
def true_u(x, t, alpha=0.05):
    return torch.sin(np.pi * x) * torch.exp(-np.pi**2 * alpha * t)

# dense grid for eval
x_test = torch.linspace(0, 1, 500)
t_test = torch.linspace(0, 1.0, 500)

X, T_grid = torch.meshgrid(x_test, t_test, indexing='ij')

# flatten for network forward pass
xt_test = torch.stack([X.flatten(), T_grid.flatten()], dim=1)

# random sparse points for regular nn
N_data = 15
x_data = torch.rand(N_data, 1)
t_data = torch.rand(N_data, 1)
xt_data = torch.cat([x_data, t_data], dim=1)
u_data = true_u(x_data, t_data)

# interior collocation points for pde residual
N = 2000

# interior region, random scatter over full (x, t) domain
x_col = torch.rand(N, 1, requires_grad=True)
t_col = torch.rand(N, 1, requires_grad=True)

# initial conditions: t = 0, random x
N_ic = 100
x_ic = torch.rand(N_ic, 1)
t_ic = torch.zeros(N_ic, 1)
u_ic = torch.sin(np.pi * x_ic) # ground truth initial condition at random x

# boundary conditions: x = 0 & x = 1, random t
N_bc = 100
t_bc = torch.rand(N_bc, 1)
x_left = torch.zeros(N_bc, 1)
x_right = torch.ones(N_bc, 1)
# u = 0 at both boundaries so target is just zeros

# model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(2, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
    def forward(self, xt):
        return self.net(xt)

# instantiate nets
nn_model = Net()
pinn_model = Net()

# optimizers
nn_opt = optim.Adam(nn_model.parameters(), lr=1e-3)
pinn_opt = optim.Adam(pinn_model.parameters(), lr=1e-3)

# mse loss
mse = nn.MSELoss()

# training
epochs = 5000

for epoch in tqdm(range(epochs)):
    
    # standard nn
    nn_opt.zero_grad()
    u_pred = nn_model(xt_data)
    loss_nn = mse(u_pred, u_data)
    loss_nn.backward()
    nn_opt.step()

    # pinn
    pinn_opt.zero_grad()

    # collocation points/physics loss on interior
    u_pred_col = pinn_model(torch.cat([x_col, t_col], dim=1))
    u_t = torch.autograd.grad(u_pred_col, t_col, grad_outputs=torch.ones_like(u_pred_col), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred_col, x_col, grad_outputs=torch.ones_like(u_pred_col), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    residual = u_t - 0.05 * u_xx # should be zero everywhere in interior
    loss_col = mse(residual, torch.zeros_like(residual))

    # enforcing initial conditions
    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_pred_ic = pinn_model(xt_ic)
    loss_ic = mse(u_pred_ic, u_ic)

    # enforcing boundary conditions
    xt_left = torch.cat([x_left, t_bc], dim=1)
    xt_right = torch.cat([x_right, t_bc], dim=1)

    u_pred_left = pinn_model(xt_left)
    u_pred_right = pinn_model(xt_right)

    loss_bc = mse(u_pred_left, torch.zeros_like(u_pred_left)) + mse(u_pred_right, torch.zeros_like(u_pred_right))

    loss_pinn = loss_col + 10*loss_ic + loss_bc
    loss_pinn.backward()
    pinn_opt.step()

# eval
with torch.no_grad():
    nn_pred = nn_model(xt_test).reshape(500,500)
    pinn_pred = pinn_model(xt_test).reshape(500,500)

u_true = true_u(X, T_grid).detach()

# plotting
x_np = x_test.numpy()
t_np = t_test.numpy()

fig, axes = plt.subplots(1, 3, figsize=(15,4))

# colorbar scaling
vmin_sol = u_true.numpy().min()
vmax_sol = u_true.numpy().max()

# ground truth
im0 = axes[0].pcolormesh(t_np, x_np, u_true.numpy(), cmap='hot', shading='auto', vmin=vmin_sol, vmax=vmax_sol)
axes[0].set_title("ground truth")
axes[0].set_xlabel('t')
axes[0].set_ylabel('x')
plt.colorbar(im0, ax=axes[0])

# nn prediction
im1 = axes[1].pcolormesh(t_np, x_np, nn_pred.numpy(), cmap='hot', shading='auto', vmin=vmin_sol, vmax=vmax_sol)
axes[1].set_title("NN (15 pts)")
axes[1].set_xlabel("t")
axes[1].set_ylabel("x")
plt.colorbar(im1, ax=axes[1])

# pinn prediction
im2 = axes[2].pcolormesh(t_np, x_np, pinn_pred.numpy(), cmap='hot', shading='auto', vmin=vmin_sol, vmax=vmax_sol)
axes[2].set_title("PINN")
axes[2].set_xlabel("t")
axes[2].set_ylabel("x")
plt.colorbar(im2, ax=axes[2])

plt.suptitle("heat equation solution: NN vs PINN")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

err_nn = (nn_pred - u_true).abs().numpy()
err_pinn = (pinn_pred - u_true).abs().numpy()

# colorbar scaling
vmax = max(err_nn.max(), err_pinn.max())

im0 = axes[0].pcolormesh(t_np, x_np, err_nn, cmap='viridis', shading='auto', vmin=0, vmax=vmax)
axes[0].set_title("|NN error|")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(t_np, x_np, err_pinn, cmap='viridis', shading='auto', vmin=0, vmax=vmax)
axes[1].set_title("|PINN error|")
plt.colorbar(im1, ax=axes[1])

plt.suptitle("absolute error")
plt.tight_layout()
plt.show()

