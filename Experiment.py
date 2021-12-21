import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
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
    noise_var = 1e-4 * np.eye(M)
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

data = pd.read_csv('student_scores.csv')
data = data.drop_duplicates(subset=['Hours'])
data = data.sort_values('Hours')
X = data['Hours'].values
y = data['Scores'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

N = 50
x = np.linspace(1, 9, N)
variance = 0.4
length = 3.0
sigma_test = Get_Kernel_Matrix(x, variance, length)
num_samples = 5

plt.figure()
mean = np.zeros(N)
ys = np.random.multivariate_normal(mean=mean, cov=sigma_test, size=num_samples)
for i in range(num_samples):
    plt.scatter(x, ys[i,:])
    plt.plot(x, ys[i,:])

plt.title('sampled functions from zero mean prior with RBF kernel (variance = {}, length = {})'.format(variance, length))
plt.show()

mu_test, sigma_test = GP(X_train, y_train, X,
                         Get_Kernel_Matrix, variance, length)
var = sigma_test.diagonal()


upper_ci = mu_test + 2*np.sqrt(var)
lower_ci = mu_test - 2*np.sqrt(var)

a_BSpline = interpolate.make_interp_spline(X, mu_test)
mu_new = a_BSpline(x)
plt.figure()
plt.scatter(X_train, y_train, label='training points')
plt.scatter(X_test, y_test, label='test points')
plt.plot(x, mu_new, linewidth=2, label='mean of the posterior')

a_BSpline = interpolate.make_interp_spline(X, upper_ci)
upper_ci = a_BSpline(x)
plt.plot(x, upper_ci)

a_BSpline = interpolate.make_interp_spline(X, lower_ci)
lower_ci = a_BSpline(x)
plt.plot(x, lower_ci)
plt.fill_between(x, upper_ci, lower_ci, facecolor='red', alpha=0.30, label='95% CI')
plt.title('Posterior with RBF kernel (variance = {}, length = {})'.format(variance, length))
plt.legend()
plt.show()


















