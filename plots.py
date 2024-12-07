import numpy as onp
import matplotlib.pyplot as plt

from sparse_gp import *

''' 
Plot sparse GP predictions with uncertainty.

Arguments:
- X_train: Training inputs.
- Y_train: Training outputs.
- X_test: Test inputs.
- Y_test: Test outputs.
- Z: Inducing points.
- optimized_hyperparams: Optimized hyperparameters.
- model_fn: Function that computes the posterior mean and variance.
- show_inducing: Whether to plot the inducing points.
'''
def plot_sparse_gp_with_uncertainty(X_train, Y_train, X_test, Y_test, Z, optimized_hyperparams, model_fn, show_inducing=True):
    # Ensure input arrays are two-dimensional
    X_test = X_test.reshape(-1, 1)
    X_train = X_train.reshape(-1, 1)
    Z = Z.reshape(-1, 1)

    # Make predictions on the test set
    posterior_mean, posterior_var = model_fn(X_test, X_train, Y_train, Z, optimized_hyperparams)

    # Compute pseudo-observations at inducing points
    inducing_mean, _ = model_fn(Z, X_train, Y_train, Z, optimized_hyperparams)

    # Convert JAX arrays to NumPy arrays
    posterior_mean = onp.array(posterior_mean).flatten()
    posterior_var = onp.array(posterior_var).flatten()
    X_test = onp.array(X_test).flatten()
    Z = onp.array(Z).flatten()
    inducing_mean = onp.array(inducing_mean).flatten()

    # Ensure variances are positive
    posterior_var = onp.maximum(posterior_var, 1e-10)

    # Plot data and predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, Y_train, color='blue', alpha=0.7, label="Training Points")
    plt.scatter(X_test, Y_test, color='red', alpha=0.7, label="Test Points")
    plt.plot(X_test, posterior_mean, 'green', label="Predicted Mean")
    plt.fill_between(X_test,
                     posterior_mean - 1.96 * onp.sqrt(posterior_var),
                     posterior_mean + 1.96 * onp.sqrt(posterior_var),
                     color='green', alpha=0.2, label="95% Confidence Interval")

    # Plot inducing points if show_inducing is True
    if show_inducing:
        plt.scatter(Z, inducing_mean, color='orange', s=100, label="Inducing Points", marker='x', zorder=5)
    #

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.title(f"Sparse GP Prediction with Uncertainty (Inducing Points: {len(Z)})")
    plt.show()
#

'''
Plot sparse GP predictions with uncertainty, excluding inducing points for real data.
Just plots the training and test data points with the predicted mean and confidence intervals.

Arguments:
- X_train: Training inputs.
- Y_train: Training outputs.
- X_test: Test inputs.
- Y_test: Test outputs.
- Z: Inducing points.
- optimized_hyperparams: Optimized hyperparameters.
- model_fn: Function that computes the posterior mean and variance.
- title_suffix: Suffix to add to the plot title.
'''
def plot_sparse_gp_with_uncertainty_clean(X_train, Y_train, X_test, Y_test, Z, optimized_hyperparams, model_fn, title_suffix=""):
    # Extract time step (assume it's the first column in X_train)
    time_train = X_train[:, 0]
    time_test = X_test[:, 0]
    
    # Make predictions for the test data
    test_posterior_mean, test_posterior_var = model_fn(X_test, X_train, Y_train, Z, optimized_hyperparams)

    # Make predictions for the training data
    train_posterior_mean, train_posterior_var = model_fn(X_train, X_train, Y_train, Z, optimized_hyperparams)

    # Ensure variances are positive
    test_posterior_var = onp.maximum(test_posterior_var, 1e-10)
    train_posterior_var = onp.maximum(train_posterior_var, 1e-10)

    # Determine y-axis limits for zooming in
    all_data_points = onp.concatenate([Y_train, Y_test, train_posterior_mean, test_posterior_mean])
    all_data_points = all_data_points[onp.isfinite(all_data_points)]
    y_min, y_max = all_data_points.min(), all_data_points.max()
    padding = 0.1 * (y_max - y_min)  # Add 10% padding for better visualization
    y_min -= padding
    y_max += padding

    # Plot
    plt.figure(figsize=(10, 6))

    # Training data points
    plt.scatter(time_train, Y_train, color='blue', alpha=0.7, label="Training Points")
    plt.plot(time_train, train_posterior_mean, 'cyan', label="Training Predicted Mean", linewidth=2)

    # Testing data points
    plt.scatter(time_test, Y_test, color='red', alpha=0.7, label="Test Points")
    plt.plot(time_test, test_posterior_mean, 'green', label="Test Predicted Mean", linewidth=2)

    # Confidence interval for test data
    plt.fill_between(
        time_test,
        test_posterior_mean - 1.96 * onp.sqrt(test_posterior_var),
        test_posterior_mean + 1.96 * onp.sqrt(test_posterior_var),
        color='green',
        alpha=0.2,
        label="95% Confidence Interval (Test)"
    )

    # Confidence interval for training data
    plt.fill_between(
        time_train,
        train_posterior_mean - 1.96 * onp.sqrt(train_posterior_var),
        train_posterior_mean + 1.96 * onp.sqrt(train_posterior_var),
        color='cyan',
        alpha=0.2,
        label="95% Confidence Interval (Train)"
    )

    # Set y-axis limits for zooming in
    plt.ylim(y_min, y_max)

    plt.xlabel("Time")
    plt.ylabel("Search Volume")
    plt.title("Sparse GP Prediction with Training and Test" + title_suffix)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    plt.grid(True)
    plt.show()
#

'''
Visualize training, test points, and inducing points for each feature on a single plot.

Arguments:
- X_train: Training feature matrix, including time steps.
- Y_train: Training target array.
- X_test: Testing feature matrix, including time steps.
- Y_test: Testing target array.
- Z: Inducing points (higher-dimensional).
- optimized_hyperparams: Optimized hyperparameters.
- model_fn: Function to compute posterior mean and variance.
'''
def plot_sparse_gp_with_uncertainty_inducing(X_train, Y_train, X_test, Y_test, Z, optimized_hyperparams, model_fn, title_suffix="", feature_names=None):
    # Ensure the time axis ranges from 0 to 800 (or determined by combined training and testing data)
    total_data_points = X_train.shape[0] + X_test.shape[0]
    time = np.arange(total_data_points)
    
    # Extract the time indices for inducing points from Z
    inducing_times = Z[:, 0]  # Assume the first column of Z represents time
    
    # Set up plot
    plt.figure(figsize=(14, 8))
    
    # Define colors for features
    num_features = X_train.shape[1]
    colors = plt.cm.get_cmap('tab10', num_features - 1)  # Exclude the first column (time)
    
    # Plot each feature
    for feature_idx in range(1, num_features):  # Skip the time column (assumed to be column 0)
        color = colors(feature_idx - 1)
        
        # Plot training points
        plt.scatter(
            time[:X_train.shape[0]], X_train[:, feature_idx],
            color=color, alpha=0.2, s=20, label=f"Training Points ({feature_names[feature_idx]})", marker='o'
        )
        
        # Plot test points
        plt.scatter(
            time[X_train.shape[0]:], X_test[:, feature_idx],
            color=color, alpha=0.2, s=20, label=f"Test Points ({feature_names[feature_idx]})", marker='s'
        )
        
        # Plot inducing points
        plt.scatter(
            inducing_times, Z[:, feature_idx],
            color=color, alpha=1.0, s=150, label=f"Inducing Points ({feature_names[feature_idx]})", marker='x'
        )
    
    # Configure plot
    plt.xlabel("Time (Index from 0 to 800)")
    plt.ylabel("Search Volume")
    plt.title("Sparse GP: Training, Test, and Inducing Points for All Features (Soccer)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False, ncol=1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#

''' 
Calculate the negative log predictive density (NLPD) for a Gaussian Process.

Arguments:
- X: Test inputs.
- Y: Test outputs.
- Z: Inducing points.
- hyperparams: Optimized hyperparameters.
- model_fn: Function that computes the posterior mean and variance.
- X_train: Training inputs.
- Y_train: Training outputs.

Returns:
- NLPD value.
'''
def calculate_nlpd(X, Y, Z, hyperparams, model_fn, X_train, Y_train):
    noise_variance = np.sum(np.array([hyperparams[i] for i in [5, 8, 10, 13, 17]]))
    posterior_mean, posterior_var = model_fn(X, X_train, Y_train, Z, hyperparams)

    # Mask valid values to exclude NaNs
    mask = np.isfinite(Y) & np.isfinite(posterior_mean) & np.isfinite(posterior_var)
    Y_filtered = Y[mask]
    posterior_mean_filtered = posterior_mean[mask]
    posterior_var_filtered = posterior_var[mask]

    nlpd = neg_log_predictive_density(Y_filtered, posterior_mean_filtered, posterior_var_filtered, noise_variance)
    return nlpd
#

'''
Calculate Mean Squared Error (MSE) for GP predictions.

Arguments:
- Y_true: True output values (test labels).
- Y_pred: Predicted mean values (from GP posterior mean).

Returns:
- mse: The Mean Squared Error value.
'''
def calculate_mse(Y_true, Y_pred):
    # Ensure the inputs are flattened
    Y_true = onp.array(Y_true).flatten()
    Y_pred = onp.array(Y_pred).flatten()

    # Mask valid values to exclude NaNs
    mask = onp.isfinite(Y_true) & onp.isfinite(Y_pred)
    Y_true_filtered = Y_true[mask]
    Y_pred_filtered = Y_pred[mask]
    
    # Compute the MSE
    mse = onp.mean((Y_true_filtered - Y_pred_filtered) ** 2)
    return mse
#

''' 
Plot the ELBO optimization process.

Arguments:
- history: List of dictionaries containing ELBO values and steps.
'''
def plot_elbo(history):
    # Extract ELBO values and steps from history
    elbo_values = [entry["elbo"] for entry in history]
    steps = [entry["step"] for entry in history]
    
    # Plot the ELBO values over steps
    plt.figure(figsize=(10, 5))
    plt.plot(steps, elbo_values, label="ELBO", color='blue')
    plt.xlabel("Step")
    plt.ylabel("ELBO")
    plt.title("ELBO Over Training Steps")
    plt.legend()
    plt.show()
#

'''
Plot kernel hyperparameters over training steps, including individual mixtures for the spectral mixture kernel.
Note: Only visualizes if there are changes in the particular hyperparameter.

Arguments:
- history: List of dictionaries containing step and hyperparameters.
- num_spectral_mixtures: Number of mixtures in the spectral mixture kernel.
- dims: Number of dimensions for the spectral mixture kernel.
''' 
def plot_kernel_hyperparameters(history, num_spectral_mixtures=10, dims=1):
    # Extract training steps.
    steps = [entry["step"] for entry in history]

    # Define associated kernels, linestyles, and colors.
    kernels = ["Squared-Exponential", "Linear", "Matern", "Sinusoidal"]
    for i in range(num_spectral_mixtures):
        kernels.append(f"Spectral Mixture {i + 1}")
    #

    linestyles = ['--', '-', '-.', ':']
    colors = ['orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'teal', 'navy']
    
    # Function to check for changes in hyperparameters
    def has_changes(values):
        return not all(v == values[0] for v in values)
    #

    # Extract and check changes in hyperparameters
    hyperparams = [        
        ("Weight (Squared-Exponential)", [entry["hyperparams"][0] for entry in history], 'blue', '--'),
        ("Weight (Linear)", [entry["hyperparams"][1] for entry in history], 'red', '--'),
        ("Weight (Matern)", [entry["hyperparams"][2] for entry in history], 'green', '--'),
        ("Weight (Sinusoidal)", [entry["hyperparams"][3] for entry in history], 'purple', '--'),
        ("Weight (Spectral Mixture)", [entry["hyperparams"][4] for entry in history], 'black', '--'),
        ("Signal Variance (Sq-Exp)", [entry["hyperparams"][6] for entry in history], 'blue', '-'),
        ("Length Scale (Sq-Exp)", [entry["hyperparams"][7] for entry in history], 'blue', ':'),
        ("Signal Variance (Linear)", [entry["hyperparams"][9] for entry in history], 'red', '-'),
        ("Signal Variance (Matern)", [entry["hyperparams"][11] for entry in history], 'green', '-'),
        ("Length Scale (Matern)", [entry["hyperparams"][12] for entry in history], 'green', ':'),
        ("Signal Variance (Sinusoidal)", [entry["hyperparams"][14] for entry in history], 'purple', '-'),
        ("Length Scale (Sinusoidal)", [entry["hyperparams"][15] for entry in history], 'purple', ':'),
        ("Period (Sinusoidal)", [entry["hyperparams"][16] for entry in history], 'purple', '--'),
    ]
    
    # Spectral Mixture kernel hyperparameters
    sm_offset = 18  # Offset for the spectral mixture kernel parameters
    for i in range(num_spectral_mixtures):
        hyperparams.append((f"SM Weight {i + 1}", [entry["hyperparams"][sm_offset + i * (1 + 2 * dims)] for entry in history], colors[i], '-'))
        for j in range(dims):
            hyperparams.append((f"SM Mean {i + 1}, Dim {j + 1}", [entry["hyperparams"][sm_offset + 1 + i * (1 + 2 * dims) + j] for entry in history], colors[i], '--'))
            hyperparams.append((f"SM Variance {i + 1}, Dim {j + 1}", [entry["hyperparams"][sm_offset + 1 + dims + i * (1 + 2 * dims) + j] for entry in history], colors[i], ':'))
        #
    #

    # Separate hyperparameters that change and those that don't
    changing_hyperparams = [(name, normalize(values), color, linestyle) for name, values, color, linestyle in hyperparams if has_changes(values)]
    static_hyperparams = [name for name, values, _, _ in hyperparams if not has_changes(values)]

    # Print unchanged hyperparameters
    if static_hyperparams:
        print("Unchanged Hyperparameters:")
        for hp_name in static_hyperparams:
            print(f" - {hp_name}")
        # 
    #

    # Plot only changing hyperparameters
    plt.figure(figsize=(12, 8))
    for name, values, color, linestyle in changing_hyperparams:
        plt.plot(steps, values, label=name, color=color, linestyle=linestyle)
    #

    # Plot configuration
    plt.xlabel("Step")
    plt.ylabel("Hyperparameter Value")
    plt.title("Kernel Hyperparameters Over Training Steps (Changing Only)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.show()
#

''' 
Function to normalize values between 0 and 1.

Arguments:
- values: List of values to normalize.

Returns:
- List of normalized values.
'''
def normalize(values):
    min_val, max_val = min(values), max(values)
    return [(v - min_val) / (max_val - min_val) for v in values] if max_val > min_val else values
#
