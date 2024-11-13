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
    noise, signal, length = hyperparams

    # Compute the pairwise squared distances
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    sq_distances = np.sum(diff ** 2, axis=-1)
    
    # Apply the squared-exponential kernel
    cov = signal * np.exp(-sq_distances / length)

    if X1.shape == X2.shape and np.all(X1 == X2):
        cov += noise * np.eye(X1.shape[0])
    #

    return cov

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
    noise, signal_variance = hyperparams

    # Compute the covariance as the dot product of X1 and X2, scaled by the signal variance
    cov = signal_variance * np.dot(X1, X2.T)

    if X1.shape == X2.shape and np.all(X1 == X2):
        cov += noise * np.eye(X1.shape[0])
    #

    return cov
#

'''
Matérn covariance function for Gaussian Processes.

Arguments:
- X1: First set of input points (shape: [N, D] for N points in D dimensions).
- X2: Second set of input points (shape: [M, D] for M points in D dimensions).
- hyperparams: List of hyperparameters [noise_variance, signal_variance, length_scale, nu].

Returns:
- Covariance matrix (shape: [N, M]).
'''
def matern_cov_function(X1, X2, hyperparams):
    noise, signal, length_scales, nu = hyperparams

    # Compute the pairwise squared distances
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    pair_matr = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Compute the Matérn covariance
    euc_term = np.sqrt(2 * nu) * pair_matr / length_scales
    euc_term = np.maximum(euc_term, 1e-16)
    
    term1 = signal * (2 ** (1 - nu) / jax.scipy.special.gamma(nu)) * (euc_term ** nu)
    bessel = kv_approx(nu, euc_term)

    cov = term1 * bessel

    if X1.shape == X2.shape and np.all(X1 == X2):
        cov += noise * np.eye(X1.shape[0])
    #
    
    return cov
#

''' 
Bessel function of the second kind (Kv) for Matérn covariance function.
Approximation for small values of x.
'''
def kv_small_x_approx(nu, x):
    term1 = gamma(nu) / 2 * (2 / x) ** nu
    term2 = 1 + (x**2) / (4 * (nu - 1))
    return term1 * term2
#

'''
Bessel function of the second kind (Kv) for Matérn covariance function.
Approximation for large values of x.
'''
def kv_large_x_approx(x):
    return np.sqrt(np.pi / (2 * x)) * np.exp(-x)
#

''' 
Bessel function of the second kind (Kv) for Matérn covariance function.
'''
def kv_approx(nu, x):
    small_x_approx = kv_small_x_approx(nu, x)
    large_x_approx = kv_large_x_approx(x)
    return np.where(x < 5, small_x_approx, large_x_approx)
#







############################################################################################################
# TO DO: Implement the following covariance functions: periodic and spectral mixture.
############################################################################################################

def sinusoidal(X1, X2, hyperparams):
    """
    sinusoidal/periodic covariance function for Gaussian Processes.

    Arguments:
    - X1: First set of input points (shape: [N, D] for N points in D dimensions).
    - X2: Second set of input points (shape: [M, D] for M points in D dimensions).
    - hyperparams: List of hyperparameters [noise_variance, signal_variance, lenth_scale, period].

    Returns:
    - Covariance matrix (shape: [N, M]).
    """
    noise, signal, length_scale, period = hyperparams
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    denominator = length_scale**2


    inner_sine = (np.pi *(distances)) / period
    numerator = -2 * (np.sin(inner_sine))**2
    cov = signal * np.exp(- (numerator/denominator))
    if X1.shape == X2.shape and np.all(X1 == X2):
       cov += noise * np.eye(X1.shape[0])
    return cov

def spectral_mixture(X1,X2, hyperparams):
    """
    Spectral Mixture covariance function for Gaussian Processes.

    Arguments:
    - X1: First set of input points (shape: [N, D] for N points in D dimensions).
    - X2: Second set of input points (shape: [M, D] for M points in D dimensions).
    - hyperparams: List of hyperparameters [weights, means, variances], where each is an array of length Q.
    - Q is the number of mixtures/spectral components we are using to represent the data

    Returns:
    - Covariance matrix (shape: [N, M]).
    """
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