import numpy as np
import matplotlib.pyplot as plt

# 1d domain
x = np.arange(-50.0, 50.0, 1.0)

# define rbf covariance kernel
def rbf_kernel(x, l):
    dist_matrix = np.abs(x[:, None] - x[None, :])
    kernel_matrix = np.exp(-dist_matrix / (2 * l ** 2))
    return kernel_matrix

test = rbf_kernel(x, 1.0)
#print(test)
#print(test.shape)

# compute lower triangular cholesky decomp
L = np.linalg.cholesky(test)
#print(L)

# draw uncorrelated noise
xi = np.random.rand(L.shape[1])
#print(xi)

# calculate field
Z = 0 + L @ xi
#print(Z)

plt.plot(Z)
plt.show()
