#!/usr/bin/env python3
"""Module of function that calculates the definiteness of a matrix"""

import numpy as np


def definiteness(matrix):
    """Function that calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.array_equal(matrix, matrix.T):
        return None
    eigenvalues, _ = np.linalg.eig(matrix)
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    return "Indefinite"
