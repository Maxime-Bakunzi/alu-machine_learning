#!/usr/bin/env python3

"""
This module with the function calculate the correlation matrix.
"""


import numpy as np


def correlation(C):
    """
    The function which calculates the correlation matrix.

    Paramenter(C):
    numpy.ndarray of shape (d, d) containing covariance matrix

    Returns:
    numpy.ndarray of shape (d, d) containing correlation matrix.
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    stddev = np.sqrt(np.diag(C))
    outer_stddev = np.outer(stddev, stddev)

    correlation_matrix = C / outer_stddev

    # Ensure the diagonal elements are exactly 1 to avoid some errors
    np.fill_diagonal(correlation_matrix, 1.0)

    return correlation_matrix
