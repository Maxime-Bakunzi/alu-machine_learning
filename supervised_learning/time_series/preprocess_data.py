#!/usr/bin/env python3
"""
preprocess_data.py
Script to preprocess raw BTC data from Coinbase and Bitstamp.

This script loads CSV files provided as command-line arguments, combines 
and sorts them by Unix time, normalizes the features (all columns except the 
timestamp) using z-score normalization, and builds sliding windows.
Each input sample consists of the past 24 hours (1440 rows) and the target 
value is the 'close' price from the row at the close of the following hour 
(60 rows after the window). Finally, the data is split into training and 
validation sets (80/20 split) and saved as 'train.npz' and 'val.npz'.
"""

import sys
import numpy as np


def load_csv(file_path):
    """
    Loads data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        np.ndarray: Array of data loaded from the CSV.
    """
    try:
        data = np.loadtxt(file_path, delimiter=',')
    except Exception as e:
        print("Error loading file {}: {}".format(file_path, e))
        sys.exit(1)
    return data


def combine_and_sort(data_list):
    """
    Combines multiple data arrays and sorts them by Unix time.

    Args:
        data_list (list): List of np.ndarray, each containing raw data.

    Returns:
        np.ndarray: Combined and sorted data array.
    """
    combined = np.vstack(data_list)
    # Sort by Unix time (first column)
    sorted_idx = np.argsort(combined[:, 0])
    return combined[sorted_idx]


def normalize_data(data):
    """
    Normalizes the data (excluding the Unix time column) using z-score 
    normalization.

    Args:
        data (np.ndarray): Raw data array.

    Returns:
        tuple: (normalized data array, mean, std) for columns 1 onwards.
    """
    features = data[:, 1:]
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    normalized = (features - mean) / std
    # Prepend the Unix time column (unchanged)
    unix_time = data[:, 0:1]
    return np.hstack((unix_time, normalized)), mean, std


def create_windows(data, window_size, target_offset):
    """
    Creates sliding windows from the data.

    Args:
        data (np.ndarray): Normalized data array.
        window_size (int): Number of rows in the input window.
        target_offset (int): Offset from the end of the window to the target 
                             row (target is the 'close' price).

    Returns:
        tuple: (X, y) where X is an array of windows and y is the array of 
               target values.
    """
    num_samples = data.shape[0] - window_size - target_offset + 1
    X = []
    y = []
    # Note: After normalization, columns 1: correspond to:
    # [open, high, low, close, BTC amount, currency, VWAP]
    # Here the 'close' price is at index 3.
    for i in range(num_samples):
        window = data[i:i + window_size, 1:]
        target_row = data[i + window_size + target_offset - 1, 1:]
        target = target_row[3]
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)


def split_data(X, y, train_ratio=0.8):
    """
    Splits the dataset into training and validation sets while preserving 
    temporal order.

    Args:
        X (np.ndarray): Array of input windows.
        y (np.ndarray): Array of target values.
        train_ratio (float): Ratio of data to be used for training.

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    split_idx = int(X.shape[0] * train_ratio)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    return X_train, y_train, X_val, y_val


def main():
    """
    Main function to load, combine, preprocess, window, split, and save data.

    Usage:
        python3 preprocess_data.py <csv_file1> <csv_file2> ...
    """
    if len(sys.argv) < 2:
        print("Usage: python3 preprocess_data.py <csv_file1> "
              "<csv_file2> ...")
        sys.exit(1)

    # Load each CSV file provided as argument
    data_list = []
    for file_path in sys.argv[1:]:
        print("Loading file: {}".format(file_path))
        data = load_csv(file_path)
        data_list.append(data)

    # Combine and sort data by Unix time
    combined_data = combine_and_sort(data_list)

    # Normalize data (Unix time remains unchanged)
    normalized_data, mean, std = normalize_data(combined_data)

    # Create windows:
    # Each window covers the past 24 hours (24*60 = 1440 rows)
    # The target is taken 60 rows after the window (the close of the following hour)
    window_size = 1440
    target_offset = 60
    X, y = create_windows(normalized_data, window_size, target_offset)

    # Split into training and validation sets
    X_train, y_train, X_val, y_val = split_data(X, y, train_ratio=0.8)

    # Save preprocessed data
    np.savez('train.npz', X=X_train, y=y_train)
    np.savez('val.npz', X=X_val, y=y_val)
    print("Preprocessed data saved to 'train.npz' and 'val.npz'.")


if __name__ == '__main__':
    main()
