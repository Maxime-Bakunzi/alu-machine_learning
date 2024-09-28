#!/usr/bin/env python3
"""
Module containing the implementation of batch normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch
     normalization.

    Args:
        Z: numpy.ndarray of shape (m, n) that should be normalized
           m is the number of data points
           n is the number of features in Z
        gamma: numpy.ndarray of shape (1, n) containing the scales used for
         batch normalization
        beta: numpy.ndarray of shape (1, n) containing the offsets used for
         batch normalization
        epsilon: small number used to avoid division by zero

    Returns:
        The normalized Z matrix
    """
    # Calculate mean of each feature
    mean = np.mean(Z, axis=0)

    # Calculate variance of each feature
    var = np.var(Z, axis=0)

    # Normalize Z
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)

    # Scale and shift
    Z_scaled = gamma * Z_norm + beta

    return Z_scaled
