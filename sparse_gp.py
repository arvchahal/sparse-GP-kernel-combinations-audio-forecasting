import jax
from jax import random
import jax.numpy as np
from jax.scipy.linalg import cho_factor,cho_solve # necessary for Cholesky factorization

from kernels import *

jax.config.update("jax_enable_x64", True)

prng_key = random.key(0)

'''
Initialize the PRNG with unique `seed`.
'''
def init_prng(seed):
    global prng_key
    prng_key = random.PRNGKey(seed)
    return prng_key
#

'''
Whenever you call random, you need to pass in as the first argument a call to this function.
This will advance the PRNG.
'''
def grab_prng():
    global prng_key
    _,prng_key = random.split(prng_key)
    return prng_key
#

'''
Transform unconstrained parameters to constrained ones.
- The first section is interpreted as unconstrained kernel weights, transformed with softmax.
- The remaining parameters are transformed to ensure positivity.

Arguments:
- unconstrained_params: Unconstrained hyperparameters.

Returns:
- constrained_params: Constrained hyperparameters (weights in [0, 1] summing to 1 and positive kernel parameters).
'''
def param_transform(unconstrained_params):
    unconstrained_params = np.array(unconstrained_params)
    
    # First part are the unconstrained weights for kernels
    unconstrained_weights = unconstrained_params[:2]  # Assuming two kernels
    
    # Apply softmax to get weights summing to 1
    weights = np.exp(unconstrained_weights) / np.sum(np.exp(unconstrained_weights))
    
    # Remaining parameters are transformed with exponentiation to ensure positivity
    other_params = np.exp(unconstrained_params[2:])
    
    # Combine weights and other transformed parameters
    constrained_params = np.concatenate([weights, other_params])
    return constrained_params
#

'''
Transform constrained parameters back to the unconstrained space.
- The weights (first 2 parameters) are converted back using log transformation.
- The remaining parameters are converted with log transformation as well.

Arguments:
- constrained_params: Constrained hyperparameters.

Returns:
- unconstrained_params: Unconstrained hyperparameters.
'''
def inverse_param_transform(constrained_params):
    constrained_params = np.array(constrained_params)
    
    # Separate weights and other parameters
    weights = constrained_params[:2]  # Assuming two kernels
    other_params = constrained_params[2:]
    
    # Convert weights with log transformation (softmax inverse)
    log_weights = np.log(weights) - np.log(weights[0])  # Normalize with respect to the first weight
    
    # Apply log transformation to the remaining parameters
    log_other_params = np.log(other_params)
    
    # Combine back into a single array
    unconstrained_params = np.concatenate([log_weights, log_other_params])
    return unconstrained_params
#

'''
Combined kernel function for Gaussian Processes, with a weighted sum of two kernels.

Arguments:
- X1: First set of input points.
- X2: Second set of input points.
- hyperparams: List of all hyperparameters [weight, noise_variance_1, signal_variance_1, length_scale_1,
                                            noise_variance_2, signal_variance_2].

Returns:
- Covariance matrix for the combined kernel.
'''
def combined_kernel(X1, X2, hyperparams):
    # Extract weights (first 2 elements)
    weights = hyperparams[:2]
    
    # Kernel hyperparameters for each individual kernel
    hyperparams_sqexp = hyperparams[2:5]  # [noise_variance_1, signal_variance_1, length_scale_1]
    hyperparams_linear = hyperparams[5:7]  # [noise_variance_2, signal_variance_2]
    
    # Compute each kernel's covariance
    K_sqexp = sqexp_cov_function(X1, X2, hyperparams_sqexp)
    K_linear = linear_cov_function(X1, X2, hyperparams_linear)
    
    # Combine with weights
    combined_cov = weights[0] * K_sqexp + weights[1] * K_linear
    return combined_cov
#

'''
Initialize inducing points as a subset of the training data.
'''
def initialize_inducing_points(X_train, num_inducing=20):
    indices = random.choice(prng_key, X_train.shape[0], shape=(num_inducing,), replace=False)
    Z = X_train[indices]
    return Z
#

'''
Compute the ELBO for sparse Gaussian processes with inducing points.

Arguments:
- combined_kernel: The combined kernel function.
- X_train: Training inputs.
- Y_train: Training outputs.
- Z: Inducing points.
- hyperparams: Hyperparameters for the combined kernel.

Returns:
- elbo: Evidence Lower Bound (ELBO) for sparse GP.
'''
def sparse_gp_elbo(combined_kernel, X_train, Y_train, Z, hyperparams):
    # Extract noise variance from hyperparameters
    noise_variance = hyperparams[1]  # Assumes first element after weight is noise variance
    
    # Compute covariance matrices
    K_XZ = combined_kernel(X_train, Z, hyperparams)
    K_ZZ = combined_kernel(Z, Z, hyperparams) + 1e-6 * np.eye(Z.shape[0])  # Jitter for numerical stability
    K_XX_diag = np.diag(combined_kernel(X_train, X_train, hyperparams))  # Diagonal of K_XX for variance
    
    # Cholesky factorization of K_ZZ
    L_ZZ, lower = cho_factor(K_ZZ, lower=True)
    
    # Intermediate calculations for ELBO
    A = cho_solve((L_ZZ, lower), K_XZ.T)
    B = np.dot(K_XZ, A) + noise_variance * np.eye(X_train.shape[0])
    
    # Cholesky factorization for B to solve for alpha
    L_B, lower_B = cho_factor(B, lower=True)
    alpha = cho_solve((L_B, lower_B), Y_train)
    
    # Data fit term
    data_fit = -0.5 * np.dot(Y_train.T, alpha)
    complexity_penalty = -0.5 * np.sum(np.log(np.diag(L_B)))
    constant_term = -0.5 * X_train.shape[0] * np.log(2 * np.pi)
    
    elbo = data_fit + complexity_penalty + constant_term
    return np.squeeze(elbo)
#

'''
Compute posterior predictive mean and variance for sparse GP with inducing points.

Arguments:
- X_star: Test inputs.
- X_train: Training inputs.
- Y_train: Training targets.
- Z: Inducing points.
- hyperparams: Hyperparameters for the combined kernel.

Returns:
- posterior_mean: Predictive mean for test points.
- posterior_var: Predictive variance for test points.
'''
def sparse_gp_posterior_predictive(X_star, X_train, Y_train, Z, hyperparams):
    # Extract noise variance from hyperparameters
    noise_variance = hyperparams[1]
    
    # Compute necessary covariance matrices
    K_XZ = combined_kernel(X_train, Z, hyperparams)  # Shape (N_train, N_inducing)
    K_ZZ = combined_kernel(Z, Z, hyperparams) + 1e-6 * np.eye(Z.shape[0])  # Shape (N_inducing, N_inducing), with jitter
    K_starZ = combined_kernel(X_star, Z, hyperparams)  # Shape (N_test, N_inducing)

    # print("K_XZ shape:", K_XZ.shape)
    # print("K_ZZ shape:", K_ZZ.shape)
    # print("K_starZ shape:", K_starZ.shape)
    
    # Cholesky factorization of K_ZZ
    L_ZZ, lower = cho_factor(K_ZZ, lower=True)
    
    # Compute A, which is used in posterior mean and variance calculations
    A = cho_solve((L_ZZ, lower), K_XZ.T)  # Shape (N_inducing, N_train)
    # print("A shape:", A.shape)
    
    # Compute B, used to solve for alpha
    B = np.dot(K_XZ, A) + noise_variance * np.eye(X_train.shape[0])  # Shape (N_train, N_train)
    # print("B shape:", B.shape)
    L_B, lower_B = cho_factor(B, lower=True)
    alpha = cho_solve((L_B, lower_B), Y_train)  # Shape (N_train,)
    # print("alpha shape:", alpha.shape)
    
    # Posterior mean calculation
    posterior_mean = np.dot(K_starZ, np.dot(A, alpha))  # Shape (N_test,)
    # print("posterior_mean shape:", posterior_mean.shape)

    # Posterior variance calculation
    v = cho_solve((L_ZZ, lower), K_starZ.T)  # Shape (N_inducing, N_test)
    posterior_cov = combined_kernel(X_star, X_star, hyperparams) - np.dot(K_starZ, v)  # Shape (N_test, N_test)
    # print("posterior_cov shape:", posterior_cov.shape)
    posterior_var = np.diag(posterior_cov) + noise_variance  # Shape (N_test,)
    # print("posterior_var shape:", posterior_var.shape)

    return posterior_mean, posterior_var
# 

'''
Compute the negative log of the predictive density, given (1) ground-truth labels Y_test, (2) the posterior mean for the test inputs,
(3) the posterior variance for the test inputs, and (4) the noise variance (to be added to posterior variance)
'''
def neg_log_predictive_density(Y_test, posterior_mean, posterior_var, noise_variance):
    # Predictive variance.
    predictive_var = posterior_var + noise_variance

    # Squared difference between the posterior and ground-truth.
    sq_diff = (Y_test - posterior_mean) ** 2

    # Negative log predictive density.
    lhs = 0.5 * np.log(2 * np.pi * predictive_var)
    rhs = sq_diff / (2 * predictive_var)

    average_neg_log_density = np.average(lhs + rhs)

    return average_neg_log_density
#

'''
Your main optimization loop.
cov_func shoud be either sqexp_cov_function or sqexp_mahalanobis_cov_function.
X_train and Y_train are the training inputs and labels, respectively.
unconstrained_hyperparams_init is the initialization for optimization.
step_size is the gradient ascent step size.
T is the number of steps of gradient ascent to take.
This function should return a 2-tuple, containing (1) the results of optimization (unconstrained hyperparameters), and
(2) the log marginal likelihood at the last step of optimization.
'''
def empirical_bayes(X_train, Y_train, Z, unconstrained_hyperparams_init, step_size, T):
    # Start with unconstrained hyperparameters
    unconstrained_hyperparams = unconstrained_hyperparams_init
    
    # Define the ELBO function with transformation applied
    elbo_func = lambda hp: sparse_gp_elbo(combined_kernel, X_train, Y_train, Z, param_transform(hp))

    # Keep track of optimized transformed hyperparameters, ELBO values, and step count
    history = []
    
    # Gradient ascent loop
    for t in range(T):
        elbo_value, grad = jax.value_and_grad(elbo_func)(unconstrained_hyperparams)
        unconstrained_hyperparams += step_size * grad

        if t % 10 == 0:
            print(f"Step {t}, ELBO: {elbo_value}")
        #

        # Store the history
            curr_history = { 
                "step": t,
                "elbo": elbo_value,
                "hyperparams": param_transform(unconstrained_hyperparams)
            }

        history.append(curr_history)
        #
    #

    # Transform to constrained space at the end
    optimized_hyperparams = param_transform(unconstrained_hyperparams)

    return optimized_hyperparams, elbo_value, history
#