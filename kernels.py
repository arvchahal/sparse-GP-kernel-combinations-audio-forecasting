import jax
import jax.numpy as np
from jax.scipy.special import gamma

'''
Squared-Exponential (RBF) covariance function for Gaussian Processes.

Arguments:
- X1: First set of input points, shape (N, D), where N is the number of points and D is the dimensionality.
- X2: Second set of input points, shape (M, D), where M is the number of points and D is the dimensionality.
- hyperparams: [noise_variance, signal_variance, length_scale]

Returns:
- Covariance matrix (shape: [N, M]).
'''
def sqexp_cov_function(X1, X2, hyperparams):
    _, signal, length = hyperparams

    # Compute the pairwise squared distances
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    sq_distances = np.sum(diff ** 2, axis=-1)
    
    # Apply the squared-exponential kernel
    cov = signal * np.exp(-sq_distances / length)

    return cov
#

'''
Linear covariance function for Gaussian Processes.

Arguments:
- X1: First set of input points (shape: [N, D] for N points in D dimensions).
- X2: Second set of input points (shape: [M, D] for M points in D dimensions).
- hyperparams: Hyperparameters [noise_variance, signal_variance].

Returns:
- Covariance matrix (shape: [N, M]).
'''
def linear_cov_function(X1, X2, hyperparams):
    _, signal_variance = hyperparams

    # Compute the covariance as the dot product of X1 and X2, scaled by the signal variance
    cov = signal_variance * np.dot(X1, X2.T)

    return cov
#

'''
Matérn covariance function for Gaussian Processes.

Arguments:
- X1: First set of input points (shape: [N, D] for N points in D dimensions).
- X2: Second set of input points (shape: [M, D] for M points in D dimensions).
- hyperparams: List of hyperparameters [noise_variance, signal_variance, length_scale].

Note: In this implementation, nu is fixed to 3/2.

Returns:
- Covariance matrix (shape: [N, M]).
'''
def matern_cov_function(X1, X2, hyperparams):
    _, signal, length_scale = hyperparams

    # Compute pairwise distances
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))  # Euclidean distance

    # Matérn kernel with nu = 3/2
    sqrt_3_r_by_l = np.sqrt(3) * r / length_scale
    cov = signal * (1 + sqrt_3_r_by_l) * np.exp(-sqrt_3_r_by_l)
    
    return cov
#

'''
Sinusoidal/periodic covariance function for Gaussian Processes.

Arguments:
- X1: First set of input points (shape: [N, D] for N points in D dimensions).
- X2: Second set of input points (shape: [M, D] for M points in D dimensions).
- hyperparams: List of hyperparameters [noise_variance, signal_variance, length_scale, period].

Returns:
- Covariance matrix (shape: [N, M]).
'''
def sinusoidal_cov_function(X1, X2, hyperparams):
    _, signal, length_scale, period = hyperparams

    # Compute the pairwise distances
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    denominator = length_scale**2

    inner_sine = (np.pi *(distances)) / period
    numerator = -2 * (np.sin(inner_sine))**2
    cov = signal * np.exp(- (numerator/denominator))

    return cov
#

'''
Spectral Mixture covariance function for Gaussian Processes.

Arguments:
- X1: First set of input points (shape: [N, D] for N points in D dimensions).
- X2: Second set of input points (shape: [M, D] for M points in D dimensions).
- hyperparams: List of hyperparameters [noise, weight0, mean0, variance0, weight1, mean1, variance1, ...].

Returns:
- Covariance matrix (shape: [N, M]).
'''
def spectral_mix_cov_function(X1,X2, hyperparams):
    # Unpack hyperparameters
    # Each should be an array of length Q except for the noise
    weights, means, variances = [], [], []
    for i in range(1, len(hyperparams), 3):
        weights.append(hyperparams[i])
        means.append(hyperparams[i+1])
        variances.append(hyperparams[i+2])
    #

    spectral_components = len(weights)
    dims = X1.shape[1]
    kernel = np.zeros((X1.shape[0], X2.shape[0]))
    
    for q in range(spectral_components):
        w_q = weights[q]
        prd = np.ones((X1.shape[0], X2.shape[0]))
        for j in range(dims):
            v_qj = variances[q][j]
            m_qj = means[q][j]
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            tau = np.sqrt(np.sum(diff ** 2, axis=-1))
            gauss = np.exp(-2*(np.pi**2)* (tau**2) / v_qj)
            cos = np.cos(2*np.pi*tau*m_qj)
            prd *= gauss*cos
        #
        kernel += prd *w_q
    #

    return kernel
#

'''
Simple Spectral Mixture covariance function for Gaussian Processes.

Arguments:
- X1: First set of input points (shape: [N, D] for N points in D dimensions).
- X2: Second set of input points (shape: [M, D] for M points in D dimensions).
- hyperparams: List of hyperparameters [weights, means, variances], where each is an array of length Q.
- Q is the number of mixtures/spectral components we are using to represent the data

Returns:
- Covariance matrix (shape: [N, M]).
'''
def simple_spectral_mixture(X1,X2, hyperparams):
    # Unpack hyperparameters
    noise, weights, means, variances = hyperparams  # Each should be an array of length Q except for the noise
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    kernel_matrix = np.zeros_like(distances)
    for q in range(len(weights)):
        w_q = weights[q]
        mu_q = means[q]
        v_q = variances[q]
        # Gaussian envelope (length scale term)
        gaussian_term = np.exp(-2 * np.pi**2 * distances**2 * v_q)
        
        # Cosine term (periodic term)
        cosine_term = np.cos(2 * np.pi * distances * mu_q)
        
        # Combine terms and accumulate in the kernel matrix
        kernel_matrix += w_q * gaussian_term * cosine_term
    if X1.shape == X2.shape and np.all(X1 == X2):
       kernel_matrix += noise * np.eye(X1.shape[0])
    return kernel_matrix
#

