import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles, make_moons
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interpolate
import math
from sklearn.metrics import pairwise_distances

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.reciprocal(1+np.exp(-x))

def RBF_kernel(x: np.ndarray,
               y: np.ndarray,
               variance = 0.8,
               length = 1.0) -> np.ndarray:
    distances = pairwise_distances(x, y, metric='sqeuclidean')
    kernel = variance * np.exp(-distances/2*length*length)
    return kernel

def likelihood(f: np.ndarray,
               y: np.ndarray) -> np.ndarray:
    yf = y*f
    return sigmoid(yf)

def log_likelihood_gradient(f: np.ndarray,
                            y: np.ndarray) -> np.ndarray:
    return (y+1)/2 - likelihood(f,y)

def log_likelihood_hessian(f: np.ndarray,
                           y: np.ndarray) -> np.ndarray:
    y1 = np.ones_like(y)
    L = likelihood(f,y1)
    return np.diag(L*(1-L))

def find_mode(y: np.ndarray,
              K: np.ndarray):
    dim = K.shape[0]
    inv_k = np.linalg.inv(K)
    f = np.zeros(dim)
    f_old = f-6
    step = 1e-1

    while np.linalg.norm(f-f_old) > 1e-5:
        # print(np.linalg.norm(f-f_old))
        grad = log_likelihood_gradient(f, y)
        W = log_likelihood_hessian(f, y)
        mat = np.linalg.inv(W+inv_k)
        vec = grad - np.matmul(inv_k, f)
        f_old = f
        f = f + step * np.matmul(mat, vec)

    return f, inv_k

def find_approx_posterior(y: np.ndarray,
                          K: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    f_hat, inv_k = find_mode(y,K)
    hessian = log_likelihood_hessian(f_hat,y)
    cov = np.linalg.inv(hessian+inv_k)
    return f_hat, cov, inv_k

def find_test_posterior(X: np.ndarray,
                        X_test: np.ndarray,
                        inv_k: np.ndarray,
                        cov_train: np.ndarray,
                        f_hat: np.ndarray,
                        variance: float,
                        length: float) -> (np.ndarray, np.ndarray):
    k_star_star = RBF_kernel(X_test, X_test, variance, length)
    k_star = RBF_kernel(X, X_test, variance, length)
    mu_star = np.matmul(k_star.T, np.matmul(inv_k, f_hat))
    temp_1 = np.matmul(inv_k, k_star)
    temp_2 = np.matmul(k_star.T, inv_k)
    cov_star = k_star_star - np.matmul(k_star.T, temp_1) + \
               np.matmul(temp_2, np.matmul(cov_train, temp_1))

    return  mu_star, cov_star

def find_mean_probability(mu_star: np.ndarray,
                          cov_star: np.ndarray) -> (np.ndarray, np.ndarray):
    num_samples = 1000
    cov_star = np.diag(np.diag(cov_star))
    samples = np.random.multivariate_normal(mu_star, cov=cov_star, size=num_samples)
    probs = sigmoid(samples)
    mean_probs = probs.mean(axis=0)
    var_probs = np.square(probs - mean_probs)
    var_probs = var_probs.mean(axis=0)
    predictions = np.ones(cov_star.shape[0])
    predictions[mean_probs < 0.5] = -1
    return mean_probs, predictions, var_probs


def generate_xor_data(num_samples):
    np.random.seed(0)
    X = np.random.randn(num_samples, 2)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

    X_pos = X[Y]
    X_neg = X[~Y]

    return X_pos, X_neg

def plot_data(pos_data, neg_data):
    plt.figure()
    plt.scatter(pos_data[:,0], pos_data[:,1], label='positive', marker='o')
    plt.scatter(neg_data[:,0], neg_data[:,1], label='negative', marker='x')
    plt.legend()
    plt.title('Training data')
    plt.show()

X_pos, X_neg = generate_xor_data(num_samples=300)
plot_data(X_pos, X_neg)
X = np.concatenate([X_pos, X_neg])
N1 = X_pos.shape[0]
N2 = X_neg.shape[0]
y_pos = np.array([1]*N1)
y_neg = np.array([-1]*N2)
y = np.concatenate([y_pos, y_neg])

variance = 1.0
length = 6.0
K = RBF_kernel(X, X, variance=variance, length=length)
K += 1e-4*np.eye(K.shape[0])
print(K.shape)

post_mean, post_cov, inv_k = find_approx_posterior(y, K)
print(post_cov.shape)

# X_test = 0.8 * np.random.randn(1000, 2) + np.array([0.0, 0.0])
X_test = np.mgrid[-3:3:0.1, -3:3:0.1].reshape(2,-1).T

mu_star, cov_star = find_test_posterior(X, X_test, inv_k, post_cov, post_mean, variance, length)

probs, preds, var_probs = find_mean_probability(mu_star, cov_star)

for i in range(len(probs)):
    print(probs[i], preds[i], var_probs[i])

indices_pos =  np.logical_and(preds == 1, var_probs < 5e-1)
# indices_pos = np.logical_and(abs(probs) < 5.1, abs(probs) > 4.9)
X_test_pos = X_test[indices_pos]
probs_pos = probs[indices_pos]
indices_neg = np.logical_and(preds == -1, var_probs < 5e-1)
X_test_neg = X_test[indices_neg]
probs_neg = probs[indices_neg]

plt.figure()

# for i in range(X_test_pos.shape[0]):
#     plt.plot(X_test_pos[i,0], X_test_pos[i,1], marker='s',
#              color=(probs_pos[i],0,1-probs_pos[i]))
#
# for i in range(X_test_neg.shape[0]):
#     plt.plot(X_test_neg[i,0], X_test_neg[i,1], marker='s',
#              color=(probs_neg[i],0,1-probs_neg[i]))

# for i in range(X_pos.shape[0]):
#     plt.plot(X_pos[i,0], X_pos[i,1], '.')
#
# for i in range(X_neg.shape[0]):
#     plt.plot(X_neg[i,0], X_neg[i,1], '.')

# plt.scatter(X_pos[:,0], X_pos[:,1], label='positive train', marker='x')
# plt.scatter(X_neg[:,0], X_neg[:,1], label='negative train', marker='o')

plt.scatter(X_test_pos[:,0], X_test_pos[:,1], marker='x', label='positive test')
# indices = np.argsort(X_test_pos[:,0])
# x = X_test_pos[:,0][indices]
# y = X_test_pos[:,1][indices]
# plt.plot(x, y, 'k-', label='decision boundary')
plt.scatter(X_test_neg[:,0], X_test_neg[:,1], marker='o', label='negative test')
# plt.title('Inference')
# plt.legend()
# plt.savefig('GPC_boundary_4.png')
plt.show()