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
def load_poltics():
    data = np.load("data.npy", allow_pickle=True)
    return data[5:10]

def load_tech():
    data = np.load("data.npy", allow_pickle=True)
    return data[0:5]

import numpy as np

def split_train_test(data, train_ratio):
    """
    Splits time series data into train and test sets based on the specified ratio,
    excluding the identifier.
    
    Arguments:
    - data: numpy array, where each row is a time series with a unique identifier followed by values.
    - train_ratio: float, the percentage of data to be used for training (e.g., 0.75 for 75% train).
    
    Returns:
    - train_data: numpy array with only the training portion of the time series values.
    - test_data: numpy array with only the testing portion of the time series values.
    """
    # Calculate the index to split each row based on the train_ratio
    split_index = lambda row: int(len(row[1:]) * train_ratio)
    
    # Initialize lists to store the split data
    train_data, test_data = [], []
    
    for row in data:
        values = row[1:]  # Only the time series values (excluding the identifier)

        # Calculate the split index for this row
        idx = split_index(row)
        
        # Split into train and test based on the calculated index
        train_data.append(values[:idx])
        test_data.append(values[idx:])
    
    # Convert lists back to numpy arrays
    train_data = np.array(train_data, dtype=object)
    test_data = np.array(test_data, dtype=object)
    
    return train_data, test_data