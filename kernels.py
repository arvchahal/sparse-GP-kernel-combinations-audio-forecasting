import jax as np
import jax
from jax import random
import jax.numpy as np
from jax.scipy.linalg import cho_factor,cho_solve # necessary for Cholesky factorization


def sqexp_cov_function(X1, X2, hyperparams):
    """
    Squared-Exponential (RBF) covariance function for Gaussian Processes.

    Arguments:
    - X1: First set of input points, shape (N, D), where N is the number of points and D is the dimensionality.
    - X2: Second set of input points, shape (M, D), where M is the number of points and D is the dimensionality.
    - hyperparams: [noise_variance, signal_variance, length_scale]
    
    Returns:
    - Covariance matrix (shape: [N, M]).
    """
    noise, signal, length = hyperparams
    delta = 0  # optional delta value for flexibility (currently unused)
    
    # Compute pairwise squared distances between points in X1 and X2
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    sq_distances = np.sum(diff ** 2, axis=-1)

    # Apply the squared-exponential kernel
    cov = signal * np.exp(-sq_distances / length)
    
    return cov
#

def linear_cov_function(X1, X2, hyperparams):
    """
    Linear covariance function for Gaussian Processes.
    
    Arguments:
    - X1: First set of input points (shape: [N, D] for N points in D dimensions).
    - X2: Second set of input points (shape: [M, D] for M points in D dimensions).
    - hyperparams: Hyperparameters [noise_variance, signal_variance].
    
    Returns:
    - Covariance matrix (shape: [N, M]).
    """
    noise_variance, signal_variance = hyperparams

    # Compute the covariance as the dot product of X1 and X2, scaled by the signal variance
    cov = signal_variance * np.dot(X1, X2.T)

    # Add noise to the diagonal if X1 and X2 are the same (for training data)
    if X1.shape == X2.shape and np.all(X1 == X2):
        cov += noise_variance * np.eye(X1.shape[0])
    #

    return cov
#