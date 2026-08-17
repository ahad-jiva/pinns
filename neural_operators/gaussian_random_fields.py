import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# 1d domain
x1 = np.arange(-50.0, 50.0, 1.0)

# 2d domain
x2 = np.linspace(0.0, 50.0, 100)
y2 = np.linspace(0.0, 50.0, 100)
X, Y = np.meshgrid(x2, y2)
points = np.stack([X.ravel(), Y.ravel()], axis=-1)

def rbf_kernel_1d(x, l):
    dist_matrix = np.abs(x[:, None] - x[None, :])
    kernel_matrix = np.exp(-dist_matrix / (2 * l ** 2))
    return kernel_matrix

def rbf_kernel_2d(points, l):
    dist_matrix = cdist(points, points, metric='euclidean')
    kernel_matrix = np.exp(-dist_matrix / (2 * l ** 2))
    return kernel_matrix

c1 = rbf_kernel_1d(x1, 1.0)

# compute lower triangular cholesky decomp
L = np.linalg.cholesky(c1)

# draw uncorrelated noise
xi = np.random.rand(L.shape[1])

# calculate field
Z = 0 + L @ xi

#plt.plot(Z)
#plt.show()

c2 = rbf_kernel_2d(points, 0.5)
L = np.linalg.cholesky(c2)
xi = np.random.rand(L.shape[1])
Z = 0 + L @ xi
print(Z.shape)
Z = np.reshape(Z, (100,100))
print(Z.shape)
print(Z)
rows, cols = Z.shape
x, y, = np.meshgrid(np.arange(cols), np.arange(rows))
plt.scatter(x.ravel(), y.ravel(), c=Z.ravel(), cmap='plasma', s=6)
plt.show()
