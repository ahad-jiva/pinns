import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

# fourth order reaction-diffusion equation
# no general closed form solution exists, so instead we construct our own and try to recover the forcing term
# u(x, t) = sin(pi x)cos(t) is a standard manufactured solution
# now we attempt to solve u_t = -u_xxxx + u_xx + f(x,t)

# ground truth
def true_u(x, t):
    return torch.sin(np.pi * x) * torch.cos(t)

# device
device = torch.device("mps" if torch.mps.is_available() else "cpu")

# dense grid for eval
x_test = torch.linspace(0, 1, 500)
t_test = torch.linspace(0, 1.0, 500)

X, T_grid = torch.meshgrid(x_test, t_test, indexing='ij')

# flatten for forward pass
xt_test = torch.stack([X.flatten(), T_grid.flatten()], dim=1).to(device)

# random sparse points for nn
N_data = 15 # give it a fighting chance
x_data = torch.rand(N_data, 1)
t_data = torch.rand(N_data, 1)
xt_data = torch.cat([x_data, t_data], dim=1).to(device)
u_data = true_u(x_data, t_data).to(device)

# interior collocation points for pde residual
N = 2500

# initial conditions: u(x,0) = sin(pi x)
N_ic = 250
x_ic = torch.rand(N_ic, 1)
t_ic = torch.zeros(N_ic, 1)
u_ic = true_u(x_ic, t_ic).to(device)

# boundary conditions (Dirichlet zero, same as heat/Langevin)
N_bc = 250
t_bc = torch.rand(N_bc, 1)
x_left = torch.zeros(N_bc, 1)
x_right = torch.ones(N_bc, 1)
# u = 0 at both boundaries so target is just zeros

# model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(2, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
    def forward(self, x):
        return self.net(x)

# instantiate nets
nn_model = Net()
pinn_model = Net()

# send to gpu
nn_model.to(device)
pinn_model.to(device)

# optimizers
nn_opt = optim.Adam(nn_model.parameters(), lr=1e-3)
pinn_opt = optim.Adam(pinn_model.parameters(), lr=1e-3)

# mse loss
mse = nn.MSELoss()

# training
epochs = 5000
pbar = tqdm(range(epochs))
for epoch in pbar:

    # standard nn
    nn_opt.zero_grad()
    u_pred = nn_model(xt_data)
    loss_nn = mse(u_pred, u_data)
    loss_nn.backward()
    nn_opt.step()

    # pinn
    pinn_opt.zero_grad()

    # resample collocation points every epoch
    x_col = torch.rand(N, 1, requires_grad=True).to(device)
    t_col = torch.rand(N, 1, requires_grad=True).to(device)

    # collocation points on interior
    u_pred_col = pinn_model(torch.cat([x_col, t_col], dim=1))
    u_t = torch.autograd.grad(u_pred_col, t_col, grad_outputs=torch.ones_like(u_pred_col), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred_col, x_col, grad_outputs=torch.ones_like(u_pred_col), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x_col, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0]
    u_xxxx = torch.autograd.grad(u_xxx, x_col, grad_outputs=torch.ones_like(u_xxx), create_graph=True)[0]

    f_col = -torch.sin(np.pi * x_col) * torch.sin(t_col) + (np.pi**4 + np.pi**2) * torch.sin(np.pi * x_col) * torch.cos(t_col)
    residual = u_t + u_xxxx - u_xx - f_col # should be zero
    loss_col = mse(residual, torch.zeros_like(residual))

    # enforcing initial conditions
    xt_ic = torch.cat([x_ic, t_ic], dim=1).to(device)
    u_pred_ic = pinn_model(xt_ic)
    loss_ic = mse(u_pred_ic, u_ic)

    # enforcing boundary conditions
    xt_left = torch.cat([x_left, t_bc], dim=1).to(device)
    xt_right = torch.cat([x_right, t_bc], dim=1).to(device)

    u_pred_left = pinn_model(xt_left)
    u_pred_right = pinn_model(xt_right)

    loss_bc = mse(u_pred_left, torch.zeros_like(u_pred_left)) + mse(u_pred_right, torch.zeros_like(u_pred_right))

    loss_pinn = loss_col + 10*loss_ic + loss_bc
    loss_pinn.backward()
    pinn_opt.step()
    
    if epoch % 20 == 0:
        pbar.set_postfix(loss_pinn=f"{loss_pinn.item():.4f}", loss_nn=f"{loss_nn.item():.4f}")

lbfgs_opt = optim.LBFGS(
    pinn_model.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=50,
    line_search_fn="strong_wolfe"
)

# fix the collocation/IC/BC points once so L-BFGS optimizes a stationary objective
x_col_lbfgs = torch.rand(N, 1, requires_grad=True).to(device)
t_col_lbfgs = torch.rand(N, 1, requires_grad=True).to(device)

pbar = tqdm(range(200))
for _ in pbar:
    def closure():
        lbfgs_opt.zero_grad()

        u_pred_col = pinn_model(torch.cat([x_col_lbfgs, t_col_lbfgs], dim=1))
        u_t = torch.autograd.grad(u_pred_col, t_col_lbfgs, grad_outputs=torch.ones_like(u_pred_col), create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred_col, x_col_lbfgs, grad_outputs=torch.ones_like(u_pred_col), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_col_lbfgs, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x_col_lbfgs, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0]
        u_xxxx = torch.autograd.grad(u_xxx, x_col_lbfgs, grad_outputs=torch.ones_like(u_xxx), create_graph=True)[0]

        f_col = -torch.sin(np.pi * x_col_lbfgs) * torch.sin(t_col_lbfgs) + \
                (np.pi**4 + np.pi**2) * torch.sin(np.pi * x_col_lbfgs) * torch.cos(t_col_lbfgs)
        residual = u_t + u_xxxx - u_xx - f_col
        loss_col = mse(residual, torch.zeros_like(residual))

        xt_ic = torch.cat([x_ic, t_ic], dim=1).to(device)
        loss_ic = mse(pinn_model(xt_ic), u_ic)

        xt_left = torch.cat([x_left, t_bc], dim=1).to(device)
        xt_right = torch.cat([x_right, t_bc], dim=1).to(device)
        loss_bc = mse(pinn_model(xt_left), torch.zeros_like(pinn_model(xt_left))) + \
                  mse(pinn_model(xt_right), torch.zeros_like(pinn_model(xt_right)))

        loss = loss_col + 10 * loss_ic + loss_bc
        loss.backward()
        return loss

    loss = lbfgs_opt.step(closure)
    pbar.set_postfix(loss=f"{loss.item():.4f}")

# eval
with torch.no_grad():
    nn_pred = nn_model(xt_test).reshape(500, 500).cpu()
    pinn_pred = pinn_model(xt_test).reshape(500, 500).cpu()

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
im1 = axes[1].pcolormesh(t_np, x_np, nn_pred.cpu().numpy(), cmap='hot', shading='auto', vmin=vmin_sol, vmax=vmax_sol)
axes[1].set_title("NN (25 pts)")
axes[1].set_xlabel("t")
axes[1].set_ylabel("x")
plt.colorbar(im1, ax=axes[1])

# pinn prediction
im2 = axes[2].pcolormesh(t_np, x_np, pinn_pred.cpu().numpy(), cmap='hot', shading='auto', vmin=vmin_sol, vmax=vmax_sol)
axes[2].set_title("PINN")
axes[2].set_xlabel("t")
axes[2].set_ylabel("x")
plt.colorbar(im2, ax=axes[2])

plt.suptitle("reaction-diffusion equation solution: NN vs PINN")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

err_nn = (nn_pred - u_true).abs().numpy()
err_pinn = (pinn_pred - u_true).abs().numpy()

mse_pinn = mse(pinn_pred, u_true).item()
mse_nn = mse(nn_pred, u_true).item()

rel_l2_pinn = (torch.norm(pinn_pred - u_true) / torch.norm(u_true)).item()
rel_l2_nn = (torch.norm(nn_pred - u_true) / torch.norm(u_true)).item()

print(f"PINN: MSE={mse_pinn:.6e}, Relative L2={rel_l2_pinn:.6e}")
print(f"NN:   MSE={mse_nn:.6e}, Relative L2={rel_l2_nn:.6e}")

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
