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
def spectral_mix_cov_function_basic(X1, X2, hyperparams):
    dims = X1.shape[1]
    num_mixtures = int((len(hyperparams) - 1) / (2 * dims + 1))

    # Extract weights, means, and variances
    weights, means, variances = [], [], []
    idx = 1  # Start after noise and num_mixtures
    for _ in range(num_mixtures):
        # Extract weight
        weights.append(hyperparams[idx])
        idx += 1

        # Extract means (dims elements)
        means.append(hyperparams[idx:idx + dims])
        idx += dims

        # Extract variances (dims elements)
        variances.append(hyperparams[idx:idx + dims])
        idx += dims
    #

    # Convert to arrays
    weights = np.array(weights)
    means = np.array(means)
    variances = np.array(variances)

    # Initialize the kernel matrix
    kernel = np.zeros((X1.shape[0], X2.shape[0]))
    for q in range(num_mixtures):
        w_q = weights[q]
        prd = np.ones((X1.shape[0], X2.shape[0]))
        for j in range(dims):
            v_qj = variances[q][j]
            m_qj = means[q][j]
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            tau = np.sqrt(np.sum(diff ** 2, axis=-1))
            gauss = np.exp(-2 * (np.pi**2) * (tau**2) / v_qj)
            cos = np.cos(2 * np.pi * tau * m_qj)
            prd *= gauss * cos
        #
        kernel += prd * w_q
    #

    return kernel


def spectral_mix_cov_function(X1, X2, hyperparams):
    # X1: (N1, D)
    # X2: (N2, D)
    # hyperparams: [noise, w_1, m_11, ..., m_1D, v_11, ..., v_1D, w_2, ...]

    dims = X1.shape[1]
    # number of mixtures
    num_mixtures = (len(hyperparams) - 1) // (2 * dims + 1)


    # Offsets for parsing hyperparams
    w_start = 1
    w_end = w_start + num_mixtures
    weights = hyperparams[w_start:w_end]  # shape: (Q,)

    m_start = w_end
    m_end = m_start + num_mixtures * dims
    means = hyperparams[m_start:m_end].reshape(num_mixtures, dims)  # shape: (Q, D)

    v_start = m_end
    v_end = v_start + num_mixtures * dims
    variances = hyperparams[v_start:v_end].reshape(num_mixtures, dims)  # shape: (Q, D)

    # Compute pairwise distances tau: (N1, N2)
    diff = X1[:, None, :] - X2[None, :, :]  # (N1, N2, D)
    tau = np.sqrt(np.sum(diff**2, axis=-1))  # (N1, N2)

    # Expand dimensions for broadcasting:
    # tau: (N1, N2, 1, 1)
    tau_expanded = tau[..., None, None]

    # means and variances: (1, 1, Q, D)
    means_expanded = means[np.newaxis, np.newaxis, :, :]
    variances_expanded = variances[np.newaxis, np.newaxis, :, :]

    # Compute gauss = exp(-2 π² τ² / v_qj)
    gauss = np.exp(-2 * (np.pi**2) * (tau_expanded**2) / variances_expanded)  # (N1, N2, Q, D)

    # Compute cos = cos(2 π τ m_qj)
    cos_terms = np.cos(2 * np.pi * tau_expanded * means_expanded)  # (N1, N2, Q, D)

    # Product over dimension j: (N1, N2, Q)
    product_over_dims = np.prod(gauss * cos_terms, axis=-1)

    # Weight each mixture and sum over Q: (N1, N2)
    kernel = np.sum(product_over_dims * weights[None, None, :], axis=-1)

    return kernel

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
