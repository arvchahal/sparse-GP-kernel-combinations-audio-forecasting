import jax.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# simple functions for loading the different data groups



#returns only the soccer team np array
def load_soccer():
    data = np.load("data.npy", allow_pickle=True)
    return data[10:]

#loads all the data 
def load_all():
    data = np.load("data.npy", allow_pickle=True)
    return data

#loads all polticial data 
def load_politics():
    data = np.load("data.npy", allow_pickle=True)
    return data[5:10]

def load_tech():
    data = np.load("data.npy", allow_pickle=True)
    return data[0:5]

import numpy as np

def split_train_test_matrix(data, train_ratio, target_column):
    """
    Splits time series data into X (features) and Y (target) for train and test sets.
    
    Arguments:
    - data: numpy array, where each row is a time series with a unique identifier followed by values.
    - train_ratio: float, the percentage of data to be used for training (e.g., 0.75 for 75% train).
    - target_column: int, index of the column in `data` to be used as Y (output target).
    
    Returns:
    - X_train: Training feature matrix (excluding target column).
    - X_test: Testing feature matrix (excluding target column).
    - y_train: Training target array (only the target column).
    - y_test: Testing target array (only the target column).
    """
    # Separate time series values, ignoring identifiers
    time_series_data = np.array([row[1:] for row in data], dtype=float)  # [num_series, num_timesteps]

    # Extract target column for Y and remove it from X
    Y = time_series_data[target_column]  # The specific row for Y (e.g., "Google" values)
    X = np.delete(time_series_data, target_column, axis=0)  # Remove target row from feature matrix

    # Transpose to shape [num_timesteps, num_series] for training/testing split
    X = X.T
    Y = Y

    # Determine split index based on the train ratio
    split_index = int(X.shape[0] * train_ratio)

    # Split X and Y into training and testing sets
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = Y[:split_index], Y[split_index:]

    return X_train, X_test, y_train, y_test



def plot_time_series(X, Y, time=None, feature_names=None, target_name="Target", title="Time Series Plot"):
    """
    Plots time series data for features and target values.

    Arguments:
    - X: numpy array, feature matrix where each column represents a feature and each row is a time step.
    - Y: numpy array, target array where each value corresponds to the target at a time step.
    - time: numpy array or list, optional, time axis for the plot (e.g., date or time index). If None, uses indices.
    - feature_names: list of strings, optional, names for each feature in X. If None, features are labeled as "Feature 0", "Feature 1", etc.
    - target_name: string, name for the target variable.
    - title: string, title of the plot.

    Returns:
    - None (displays the plot).
    """
    if time is None:
        time = np.arange(X.shape[0])  # Default to indices if no time is provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]  # Default feature names

    # Plot each feature in X
    for i in range(X.shape[1]):
        plt.plot(time, X[:, i], label=feature_names[i], linestyle='--')

    # Plot the target variable Y
    plt.plot(time, Y, label=target_name, linewidth=2, color='black')

    # Add plot labels and legend
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()