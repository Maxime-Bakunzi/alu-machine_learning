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
    # Use reshape and transpose without explicitly checking for an empty matrix
    # If the matrix is empty, reshape will raise a ValueError, which we handle
    try:
        matrix_2d = matrix.reshape(-1, matrix.shape[-1])
        transposed_matrix = matrix_2d.T
        return transposed_matrix.reshape(matrix.shape[::-1])
    except ValueError:
        return matrix
