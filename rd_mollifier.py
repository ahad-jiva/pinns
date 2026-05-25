import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

# fourth order reaction diffusion again, but with a mollifier layer instead of autodiff

# ground truth
def true_u(x, t):
    return torch.sin(np.pi * x) * torch.cos(t)

# dense grid for eval
x_test = torch.linspace(0, 1, 500)
t_test = torch.linspace(0, 1.0, 500)

X, T_grid = torch.meshgrid(x_test, t_test, indexing='ij')

# flatten for foward pass
xt_test = torch.stack([X.flatten(), T_grid.flatten()], dim=1)

# not implementing regular nn for this case

# uniform grid for mollifier based derivatives
N_x = 2000
x_grid = torch.linspace(0.0, 1.0, N_x).view(-1, 1)
hx = (x_grid[1] - x_grid[0]).item()

# time samples per epoch
N_t = 16

# points for ic/bc losses
N_ic = 257
x_ic = x_grid.clone()
t_ic = torch.zeros_like(x_ic)
u_ic = true_u(x_ic, t_ic)

N_bc = 128
t_bc = torch.rand(N_bc, 1)
x_left = torch.zeros(N_bc, 1)
x_right = torch.ones(N_bc, 1)

# model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(2, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
    def forward(self, x):
        return self.net(x)

# mollifier layer
class MollifierLayer(nn.Module):
    def __init__(self, epsilon=0.08, n_kernel=81):
        super().__init__()
        # kernel grid centered at zero
        s = torch.linspace(-epsilon, epsilon, n_kernel)
        ds = (s[1] - s[0]).item()

        # analytic derivatives via finite difference
        arg = 1.0 - (s / epsilon) ** 2
        phi = torch.where(arg > 1e-12, torch.exp(-1.0 / arg.clamp(min=1e-12)), torch.zeros_like(s))

        # normalize to integrate to 1
        phi = phi / (phi.sum() * ds)

        phi_np = phi.numpy()

        # 4th derivatives
        d1 = np.gradient(phi_np, ds)
        d2 = np.gradient(d1, ds)
        d3 = np.gradient(d2, ds)
        d4 = np.gradient(d3, ds)

        # register kernels as buffers since they dont get trained
        self.register_buffer('phi0', torch.tensor(phi_np, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer('phi2', torch.tensor(d2, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer('phi4', torch.tensor(d4, dtype=torch.float32).view(1, 1, -1))
        self.radius = n_kernel // 2
        self.n_kernel = n_kernel
    
    def forward(self, g_spatial, hx):

        pad = self.radius
        g = g_spatial.view(1, 1, -1)
        u = F.conv1d(g, self.phi0, padding=pad) * hx
        u_xx = F.conv1d(g, self.phi2, padding=pad) * hx
        u_xxxx = F.conv1d(g, self.phi4, padding=pad) * hx
        return u[0, 0], u_xx[0, 0], u_xxxx[0, 0]

# instantiate net and mollifier layer
g_model = Net()
mollifier = MollifierLayer(epsilon=0.1, n_kernel=201)

# optimizers
opt = optim.Adam(g_model.parameters(), lr=1e-3)

# mse loss
mse = nn.MSELoss()

# training
epochs = 5000

for epoch in tqdm(range(epochs)):

    opt.zero_grad()

    # random time batch, fixed x-grid
    t_batch = torch.rand(N_t, 1)

    pde_losses = []
    bc2_losses = []

    for k in range(N_t):
        t_val = t_batch[k:k+1]
        t_grid = t_val.expand(N_x, 1).clone().requires_grad_(True)

        xt = torch.cat([x_grid, t_grid], dim=1)
        g_pred = g_model(xt).squeeze(-1)

        u_pred, u_xx, u_xxxx = mollifier(g_pred, hx)

        u_t = torch.autograd.grad(u_pred, t_grid, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0].squeeze(-1)


        f_col = (-torch.sin(np.pi * x_grid.squeeze(-1)) * torch.sin(t_val.squeeze()) + (np.pi**4 - np.pi**2) * torch.sin(np.pi * x_grid.squeeze(-1)) * torch.cos(t_val.squeeze()))

        r = mollifier.radius
        residual = u_t[r:-r] - (-u_xxxx[r:-r] + u_xx[r:-r] + f_col[r:-r])
        pde_losses.append(torch.mean(residual**2))

        # extra BC for fourth-order operator: u_xx = 0 at boundaries for this manufactured solution
        bc2_losses.append(u_xx[0]**2 + u_xx[-1]**2)

    loss_col = torch.stack(pde_losses).mean()
    loss_bc2 = torch.stack(bc2_losses).mean()

    # initial condition: use mollified u, not raw g
    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    g_ic = g_model(xt_ic).squeeze(-1)
    u_ic_pred, _, _ = mollifier(g_ic, hx)
    loss_ic = mse(u_ic_pred.view(-1, 1), u_ic)

    # Dirichlet BC: use mollified u, not raw g
    xt_left = torch.cat([x_left, t_bc], dim=1)
    xt_right = torch.cat([x_right, t_bc], dim=1)

    g_left = g_model(xt_left).squeeze(-1)
    g_right = g_model(xt_right).squeeze(-1)

    u_left, _, _ = mollifier(g_left, 1.0)
    u_right, _, _ = mollifier(g_right, 1.0)

    loss_bc = torch.mean(u_left**2) + torch.mean(u_right**2)

    loss = 10*loss_col + 10*loss_ic + 10*loss_bc + loss_bc2
    loss.backward()
    opt.step()

# eval
with torch.no_grad():
    U_pred = []

    for j in range(len(t_test)):
        t_val = t_test[j:j+1]
        t_grid = t_val.expand(len(t_test), 1)
        xt = torch.cat([x_test.view(-1, 1), t_grid], dim=1)

        g_pred = g_model(xt).squeeze(-1)
        u_pred, _, _ = mollifier(g_pred, (x_test[1] - x_test[0]).item())
        U_pred.append(u_pred)

    pinn_pred = torch.stack(U_pred, dim=1)

u_true = true_u(X, T_grid).detach()

# plotting
x_np = x_test.numpy()
t_np = t_test.numpy()

fig, axes = plt.subplots(1, 2, figsize=(10,4))

# colorbar scaling
vmin_sol = u_true.numpy().min()
vmax_sol = u_true.numpy().max()

# ground truth
im0 = axes[0].pcolormesh(t_np, x_np, u_true.numpy(), cmap='hot', shading='auto', vmin=vmin_sol, vmax=vmax_sol)
axes[0].set_title("ground truth")
axes[0].set_xlabel('t')
axes[0].set_ylabel('x')
plt.colorbar(im0, ax=axes[0])

# pinn prediction
im1 = axes[1].pcolormesh(t_np, x_np, pinn_pred.numpy(), cmap='hot', shading='auto', vmin=vmin_sol, vmax=vmax_sol)
axes[1].set_title("PINN + Mollifier")
axes[1].set_xlabel("t")
axes[1].set_ylabel("x")
plt.colorbar(im1, ax=axes[1])

plt.suptitle("reaction-diffusion equation solution: PINN + Mollifier Layer")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(5, 4))

err_pinn = (pinn_pred - u_true).abs().numpy()

im0 = axes.pcolormesh(t_np, x_np, err_pinn, cmap='viridis', shading='auto')
axes.set_title("|PINN error|")
plt.colorbar(im0, ax=axes)

plt.suptitle("absolute error")
plt.tight_layout()
plt.show()
