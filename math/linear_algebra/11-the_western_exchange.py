#!/usr/bin/env python3
"""
This module defines a function transpose matrix.
"""

def np_transpose(matrix):
    """
    Transposes a numpy.ndarray.

    Args:
        matrix: A numpy.ndarray.

    Returns:
        A new numpy.ndarray which is the transpose of the input matrix.
    """
    # Return the transpose of the input matrix
    return matrix.reshape(matrix.shape[::-1])
