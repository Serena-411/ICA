#小组成员：宋若芊202000700058,张以欣202000700012,鞠颜鸿202000700048,曾飞飏202000700072
# Some comments are at the end of the code.
import numpy as np
import time

# Set a random seed for reproducibility
np.random.seed(51)

# define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# define probability density function p(x)
def p(x):
    return sigmoid(x) * (1 - sigmoid(x))

# pre-processing 1: center it to have zero mean
def centering(matrix):
    mean = np.mean(matrix, axis=1, keepdims=True)
    centered_matrix = (matrix - mean)
    return centered_matrix

# pre-processing 2: whiten it to have identity covariance
def whitening(matrix):
    cov = np.cov(matrix)
    U, S, V = np.linalg.svd(cov)
    D = np.diag(1.0 / np.sqrt(S))
    whitened_matrix = np.dot(np.dot(np.dot(U, D), U.T), matrix)
    return whitened_matrix

# set 2-by-m signal matrix S
m = 500                                 # number of samples
s1 = np.random.laplace(0,1,m)           # Laplace signal
initial_samples = np.linspace(0,1,m)    # generate m samples in [0,1]
weights = p(initial_samples) / np.sum(p(initial_samples)) # calculate weight for each sample(the sum of p must equal to 1)
s2 = np.random.choice(initial_samples, size=m, replace=True, p=weights) # Sigmoid signal
S_original = np.array([s1,s2])
S_center = centering(S_original)        # centering S
S = whitening(S_center)                 # whitening S

# pick a 2-by-2 mixing matrix A
A = np.array([[1, 0], [0, 1]])
# calculate m recording samples (matrix X)
X = np.dot(A,S)

# define ICA algorithm
def ica(X, error_tolerance=0.01, epoch=100, learning_rate=0.1, batch_size=1):
    # initialize matrix W
    n = X.shape[0]
    W = np.random.rand(n,n)

    for i in range(epoch):
        # select 1-simple mini batch randomly
        idx = np.random.randint(X.shape[1], size=batch_size)
        x_batch = X[:, idx]
        # update method
        g_derivative = -2 * np.tanh(np.dot(W, x_batch))
        det_W = np.linalg.det(W)
        if abs(det_W) < 1e-8:
            raise ValueError('weight matrix is not invertible')
        delta_W = (np.dot(g_derivative, x_batch.T) + np.linalg.inv(W.T))
        # update W
        W += learning_rate * delta_W
        # determine if it converges
        if np.max(np.abs(delta_W)) < error_tolerance:
            return np.linalg.inv(W)
    return np.linalg.inv(W)

# parameters
epoch_range = range(500, 1800, 100)
learning_rate_range = np.arange(0, 0.101, 0.005)
error_tolerance = 0.001
batch_size = 10

# try all parameters to find the suitable ones
min_error = float('inf')
best_epoch = 0
best_lr = 0
start_time = time.time()
for learning_rate in learning_rate_range:
    for epoch in epoch_range:
        A_ica = ica(X, learning_rate=learning_rate, epoch=epoch)
        error = np.linalg.norm(A_ica - A)
        if error < min_error:
            min_error = error
            best_A = A_ica.copy()
            best_epoch = epoch
            best_lr = learning_rate
total_time = time.time() - start_time

print("my program finished in:{}".format(total_time))
print("Parameters are:")
print("Laplace signal and Sigmoid signal")
print("{} each".format(m))
print("learning rate is:{}".format(best_lr))
print("error tolerance is:{}".format(error_tolerance))
print("I use {}-sample mini-batch in each of total {} epochs.".format(batch_size,best_epoch))
print("The original mixing matrix is: [[1.0, 0.0], [0.0, 1.0]]")
print("Now training and printing the best mixing matrix so far: ... ... ...")
print(best_A)

'''
comments:
when the training should stop:
Set the max iterations. When the time is up to the max iteration, the training should stop.
Set the error tolerance. When the error is less than the tolerance, the training should stop.

how to measure one approximation of A is better than another:
Iterates the parameters in the ICA algorithm, including epoch and learning rate.
Determine the best parameter and A by calculating the error of the training value and the true value of A.
'''