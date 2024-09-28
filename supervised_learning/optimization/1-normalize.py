#!/usr/bin/env python3
"""Module for normalizing a matrix"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Args:
    X (numpy.ndarray): Matrix of shape (d, nx) to normalize
        d is the number of data points
        nx is the number of features
    m (numpy.ndarray): Array of shape (nx,) that contains the mean of all
     features of X
    s (numpy.ndarray): Array of shape (nx,) that contains the standard
     deviation of all features of X

    Returns:
    numpy.ndarray: The normalized X matrix
    """
    return (X - m) / s
