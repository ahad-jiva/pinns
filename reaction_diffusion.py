import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

# fourth order reaction-diffusion equation

# ground truth
def true_u():
    return # implement ts

# dense grid for eval
# ...

# random sparse points for nn
N_data = 25 # give it a fighting chance
# ...

# interior collocation points for pde residual
N = 2000

# initial conditions
# ...

# boundary conditions
# ...

# model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(INPUTS, 32),
                nn.Tanh(),
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

# mse loss
mse = nn.MSELoss()

# training
epochs = 5000
