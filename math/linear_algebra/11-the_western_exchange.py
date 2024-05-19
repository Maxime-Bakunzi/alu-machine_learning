#!/usr/bin/env python3
"""
This module defines a function transpose matrix.
"""


def np_transpose(matrix):
    """
    Transpose the given matrix.

    Args:
        matrix (numpy.ndarray): The input matrix.

    Returns:
        numpy.ndarray: The transposed matrix.
    """
    # Reshape the matrix to 2D if possible, otherwise return the matrix itself
    try:
        matrix_2d = matrix.reshape(-1, matrix.shape[-1])
    except ValueError:
        return matrix

    # Transpose the 2D matrix and reshape it back to the original shape
    transposed_matrix = matrix_2d.T
    return transposed_matrix.reshape(matrix.shape[::-1])
