#!/usr/bin/env python3
"""
The Module of converting numeric label into one-hot matrix
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
        Y (numpy.ndarray): numeric class labels with shape (m,)
        classes (int): the maximum number of classes found in Y

    Returns:
        A one-hot encoding of Y with shape (classes, m), or None on failure.
    """

    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None

    try:
        # Initialize a zero matrix with the shape (classes, m)
        one_hot_matrix = np.zeros((classes, Y.shape[0]))

        # Set the corresponding class index to 1
        one_hot_matrix[Y, np.arange(Y.shape[0])] = 1

        return one_hot_matrix
    except Exception:
        return None
