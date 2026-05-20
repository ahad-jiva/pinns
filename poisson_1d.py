import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

# ground truth function
def true_u(x):
    return torch.sin(3 * np.pi * x)

# dense grid, to be used for eval
x_test = torch.linspace(0, 2, 400).view(-1, 1)
u_test = true_u(x_test)

# sparse training data
x_train = torch.rand(4,1)
u_train = true_u(x_train)

# collocation points for pinn error calculation
x_phys = torch.linspace(0, 2, 300).view(-1, 1)
x_phys.requires_grad = True

# model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(1, 32),
                nn.Tanh(),  # for smoothness
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
    def forward(self, x):
        return self.net(x)

# instantiate nets
nn_model = Net()
pinn_model = Net()

# optimizers
nn_opt = optim.Adam(nn_model.parameters(), lr=1e-3)
pinn_opt = optim.Adam(pinn_model.parameters(), lr=1e-3)

mse = nn.MSELoss()

# training
epochs = 5000

x_bc = torch.tensor([[0.0], [3.0]])
u_bc = torch.tensor([[0.0], [0.0]])

for epoch in tqdm(range(epochs)):

    # standard network
    nn_opt.zero_grad()
    u_pred = nn_model(x_train)
    loss_nn = mse(u_pred, u_train)
    loss_nn.backward()
    nn_opt.step()

    # pinn
    pinn_opt.zero_grad()

    # data loss (mse of same sparse training points)
    u_pred_data = pinn_model(x_train)
    loss_data = mse(u_pred_data, u_train)

    # physics loss (physics equation constraint)
    u_pred_phys = pinn_model(x_phys)

    # evaluating pde with autodiff
    grad1 = torch.autograd.grad(u_pred_phys, x_phys, grad_outputs=torch.ones_like(u_pred_phys), create_graph=True)[0]
    grad2 = torch.autograd.grad(grad1, x_phys, grad_outputs=torch.ones_like(grad1), create_graph=True)[0]

    f = ((3 * np.pi) ** 2) * torch.sin(3 * np.pi * x_phys)
    loss_phys = mse(-grad2, f)

    # enforcing boundary conditions
    loss_bc = mse(pinn_model(x_bc), u_bc)

    loss_pinn = 10*loss_data + loss_phys + 10*loss_bc
    # loss_pinn = loss_phys + loss_bc
    loss_pinn.backward()
    pinn_opt.step()

# eval
nn_pred = nn_model(x_test).detach()
pinn_pred = pinn_model(x_test).detach()

# plots
plt.figure()
plt.plot(x_test, u_test, label="Ground truth")
plt.plot(x_test, nn_pred, "--", label="NN (5pts)")
plt.plot(x_test, pinn_pred, "-", label="PINN")
plt.scatter(x_train, u_train, color='red', label="Training points")
plt.legend()
plt.title("NN vs PINN on 1d poisson")
plt.show()
