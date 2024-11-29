import jax.numpy as jnp
import numpy as np
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
import pandas as pd
import matplotlib.pyplot as plt


def load_data(group_slice):
    """
    Loads the data from 'data.npy' and extracts the specified group.
    Arguments:
    - group_slice: slice object, specifies the range of rows to load.

    Returns:
    - data: numpy array, the extracted data group.
    """
    data = np.load("data.npy", allow_pickle=True)
    return data[group_slice]


def extract_names_and_values(group):
    """
    Extracts names and numerical values from a data group.
    Arguments:
    - group: numpy array, where the first column contains names and the rest contains numeric data.

    Returns:
    - names: list, names extracted from the first column.
    - values: numpy array, numeric data extracted from the remaining columns.
    """
    names = [row[0].split('_en')[0] for row in group]  # Extract names before '_en'
    values = np.array([row[1:].astype(float) for row in group])  # Convert to float
    return names, values


def smooth_data(values, window_size=31):
    """
    Applies sliding median smoothing to each row of the data.
    Handles constant, sparse, or short time series gracefully.
    """
    smoothed_values = np.empty_like(values, dtype=float)

    for i in range(values.shape[0]):
        row = values[i]
        if np.nanstd(row) == 0:  # Check if row is constant
            smoothed_values[i] = row  # Skip smoothing
            continue

        effective_window_size = min(window_size, len(row))  # Adjust for short series
        half_window = effective_window_size // 2
        smoothed_row = []

        for j in range(len(row)):
            start = max(0, j - half_window)
            end = min(len(row), j + half_window + 1)
            window = row[start:end]
            window = window[~np.isnan(window)]  # Exclude NaN values

            if len(window) > 0:
                smoothed_row.append(np.median(window))
            else:
                smoothed_row.append(np.nan)

        smoothed_values[i] = smoothed_row

    return smoothed_values


def plot_group(names, smoothed_values, dates, title):
    """
    Plots a group of time series data.
    Arguments:
    - names: list, names of the time series.
    - smoothed_values: numpy array, smoothed time series data.
    - dates: list or numpy array, time axis for the plot.
    - title: string, title of the plot.

    Returns:
    - None (displays the plot).
    """
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(names):
        plt.plot(dates, smoothed_values[i], label=name)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Views")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


def split_train_test_matrix(data, train_ratio, target_column):
    """
    Splits time series data into X (features) and Y (target) for train and test sets.
    Includes a number line (time step indices) in the features.
    
    Arguments:
    - data: numpy array, where each row is a time series with a unique identifier followed by values.
    - train_ratio: float, the percentage of data to be used for training (e.g., 0.75 for 75% train).
    - target_column: int, index of the column in `data` to be used as Y (output target).
    
    Returns:
    - X_train: Training feature matrix (excluding target column but including time step indices).
    - X_test: Testing feature matrix (excluding target column but including time step indices).
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

    # Create the number line (time step indices)
    time_steps = np.arange(1, X.shape[0] + 1).reshape(-1, 1)  # Shape [num_timesteps, 1]

    # Add time steps as a new column to X
    X = np.hstack((time_steps, X))  # Shape [num_timesteps, num_features + 1]

    # Determine split index based on the train ratio
    split_index = int(X.shape[0] * train_ratio)

    # Split X and Y into training and testing sets
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = Y[:split_index], Y[split_index:]

    return X_train, X_test, y_train, y_test

def convert_to_numeric(arr):
    """
    Converts array elements to numeric, replacing non-numeric values with NaN.
    """
    return np.array([pd.to_numeric(x, errors='coerce') for x in arr])




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
        feature_names = [f"Feature {i - 1}" for i in range(1, X.shape[1])]  # Default feature names

    # Plot each feature in X
    for i in range(1, X.shape[1]):
        plt.plot(time, X[:, i], label=feature_names[i], linestyle='--')

    # Plot the target variable Y
    plt.plot(time, Y, label=target_name, linewidth=2, color='purple')

    # Add plot labels and legend
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
#