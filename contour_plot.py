import numpy as np
import os
from matplotlib import pyplot as plt

mu = np.array([[5], [5]])
sigma = np.array([[2,1],[1,3]])

def BiVariateNormal(x: np.ndarray) -> float:
    distance = np.matmul(np.matmul((x-mu).T, np.linalg.inv(sigma)), x-mu)
    distance = np.asscalar(distance)
    det = np.linalg.det(sigma)
    scaling = np.sqrt(2*np.pi*det**2)
    z = np.exp(-distance)/scaling
    return z

x = np.array([[5], [5]])

x = np.linspace(2, 8, 100)
X, Y = np.meshgrid(x,x)
xx = np.ravel(X)
yy = np.ravel(Y)
input = np.vstack((xx,yy))
Z = [BiVariateNormal(input[:,i].reshape(-1,1)) for i in range(len(xx))]
Z = np.array(Z).reshape(X.shape)

plt.contourf(X, Y, Z, cmap='Blues')
plt.title('Contour levels')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()
plt.show()
