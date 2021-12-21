import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interpolate
import math
from sklearn.metrics import pairwise_distances

np.random.seed(42)

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

mean = np.zeros(2)
N1 = 300
N2 = 300

pos_cov = 1.0*np.eye(2)
neg_cov = 1.0*np.eye(2)

pos = np.random.multivariate_normal(mean+2, cov=pos_cov, size=N1)
neg = np.random.multivariate_normal(mean-2, cov=neg_cov, size=N2)
print(pos.shape, neg.shape)
X = np.concatenate([pos, neg])
print(X.shape)

y_pos = np.array([1]*N1)
y_neg = np.array([-1]*N2)
y = np.concatenate([y_pos, y_neg])
print(y.shape)

plt.figure()
plt.scatter(pos[:,0], pos[:,1], label='positive', marker='x')
plt.scatter(neg[:,0], neg[:,1], label='negative', marker='o')
plt.title('Training samples')
plt.legend()
#plt.savefig('GPC_training_data.png')
plt.show()

variance = 1.0
length = 1.5
K = RBF_kernel(X, X, variance=variance, length=length)
K += 1e-3*np.eye(K.shape[0])
print(K.shape)

post_mean, post_cov, inv_k = find_approx_posterior(y, K)
print(post_cov.shape)

X_test = np.mgrid[-2.5:2.5:0.1, -2.5:2.5:0.1].reshape(2,-1).T

mu_star, cov_star = find_test_posterior(X, X_test, inv_k, post_cov, post_mean, variance, length)

probs, preds, var_probs = find_mean_probability(mu_star, cov_star)

#for i in range(len(probs)):
#    print(probs[i], preds[i], var_probs[i])

indices =  np.logical_and(abs(probs) < 0.51, abs(probs) > 0.49)
X_test_pos = X_test[indices]

plt.figure()
plt.scatter(pos[:,0], pos[:,1], label='positive train', marker='x')
plt.scatter(neg[:,0], neg[:,1], label='negative train', marker='o')
plt.scatter(X_test_pos[:,0], X_test_pos[:,1], marker='x')
indices = np.argsort(X_test_pos[:,0])
x = X_test_pos[:,0][indices]
y = X_test_pos[:,1][indices]
plt.plot(x, y, 'k', label='decision boundary', linewidth=2)
plt.title('Inference')
plt.legend()
#plt.savefig('GPC_boundary{}.png'.format(length))
plt.show()
