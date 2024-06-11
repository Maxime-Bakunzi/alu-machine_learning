#!usr/bin/env python3
"""
This will calculate the mean and covariance
"""


import numpy as np


def mean_cov(X):
    """
    The function to calculate the mean and covariance.
    """

    # X = np.ndarray(n, d)

    n, d = X.shape

    if X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)

    cov = ((X - mean).T @ (X - mean)) / (n - 1)

    return mean, cov
