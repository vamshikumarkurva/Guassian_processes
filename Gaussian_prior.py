import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interpolate

def RBF_kernel(x: np.ndarray, y: np.ndarray,
               variance = 1.0, length = 0.8 ) -> float:
    distance = - np.linalg.norm(x-y)**2/(2*length*length)
    sim = variance* np.exp(distance)
    return sim

def Get_Kernel_Matrix(x: np.ndarray, variance, length):
    X, Y = np.meshgrid(x, x)
    xx = np.ravel(X)
    yy = np.ravel(Y)
    input = np.vstack((xx, yy))
    Z = [RBF_kernel(input[0, i], input[1, i],
                    variance=variance,
                    length=length) for i in range(len(xx))]

    Z = np.array(Z).reshape(X.shape)
    return Z

def GP(X_train, Y_train, X_test, kernel_func, variance, length):
    M = len(X_train)
    N = len(X_test)

    X = np.concatenate([X_train, X_test])
    Sigma = kernel_func(X, variance, length)
    noise_var = 1e-8 * np.eye(M)
    Sigma[:M, :M] += noise_var

    sigma_1 = Sigma[:M, :M]
    sigma_2 = Sigma[M:, M:]
    sigma_12 = Sigma[:M, M:]
    sigma_21 = sigma_12.T

    mu1 = np.zeros(M)
    mu2 = np.zeros(N)
    mu1 += np.mean(Y_train)
    mu2 += np.mean(Y_train)

    factor = np.matmul(sigma_21, np.linalg.inv(sigma_1))
    mu_21 = mu2 + np.matmul(factor, Y_train - mu1)
    sigma_21 = sigma_2 - np.matmul(factor, sigma_12)

    return mu_21, sigma_21

N = 20
x = np.linspace(-5,5,N)
variance = 0.8
length = 0.2

Z = Get_Kernel_Matrix(x, variance, length)

ax = sns.heatmap(Z, cmap='Blues', vmin=0.0, vmax=1.0)
plt.axis('off')
plt.show()

mean = np.zeros(N)
num_functions = 5

ys = np.random.multivariate_normal(mean=mean, cov=Z, size=num_functions)

title = 'Functions sampled from RBF kernel with variance = {}, length = {}'.format(variance, length)
plt.title(title)
plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.ylim((-5, 5))
x1 = np.linspace(-5,5,50)
y1 = np.zeros(50)+2*np.sqrt(variance)
y2 = np.zeros(50)-2*np.sqrt(variance)
plt.scatter(x1, y1, marker='_', c='b', label='$\mu+2\sigma$')
plt.scatter(x1, y2, marker='_', c='b', label='$\mu-2\sigma$')
plt.plot(x1, np.zeros(50), label='$\mu$', linewidth=2.0)
plt.fill_between(x1, y1, y2, facecolor='red', alpha=0.30, label='95% CI')

x_new = np.linspace(-5,5,200)
for i in range(num_functions):
    a_BSpline = interpolate.make_interp_spline(x, ys[i,:])
    y_new = a_BSpline(x_new)
    plt.scatter(x, ys[i,:])
    plt.plot(x_new, y_new)

plt.legend()
plt.show()

X2 = x
N = len(X2)

X1 = np.array([X2[4], X2[8], X2[12], X2[14]])
M = len(X1)
y1 = np.array([2.4, 1.6, 1.2, 1.0])
mu1 = np.zeros(M)

mu_21, sigma_21 = GP(X1, y1, X2, Get_Kernel_Matrix, variance, length)

plt.figure(1)
plt.scatter(X1, y1, label='training data')

num_functions = 2

ys = np.random.multivariate_normal(mean=mu_21, cov=sigma_21, size=num_functions)
x_new = np.linspace(-5,5,200)

for i in range(num_functions):
    a_BSpline = interpolate.make_interp_spline(X2, ys[i, :])
    y_new = a_BSpline(x_new)
    plt.plot(x_new, y_new, linestyle='dashed', label='sample{}'.format(i))

var = sigma_21.diagonal()
upper_ci = mu_21 + np.sqrt(var)*2
a_BSpline = interpolate.make_interp_spline(X2, upper_ci)
upper_ci = a_BSpline(x_new)
plt.plot(x_new, upper_ci, 'r')

lower_ci = mu_21 - np.sqrt(var)*2
a_BSpline = interpolate.make_interp_spline(X2, lower_ci)
lower_ci = a_BSpline(x_new)
plt.plot(x_new, lower_ci, 'r')

plt.fill_between(x_new, upper_ci, lower_ci, facecolor='red', alpha=0.30, label='95% confidence interval')

a_BSpline = interpolate.make_interp_spline(X2, mu_21)
mean_new = a_BSpline(x_new)
plt.plot(x_new, mean_new, label='mean', linewidth=2.0)
plt.legend()
plt.title('Samples from posterior (RBF with variance = {}, length = {})'
          'after observing some training samples'.format(variance, length))
plt.xlabel('x')
plt.ylabel('y = f(x)')

plt.show()









