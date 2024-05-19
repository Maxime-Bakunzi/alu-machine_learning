#!/usr/bin/env python3
"""
This module defines a function to find shape of an array.
"""


def np_shape(matrix):
    """
    Calculate the shape of a nested list.

    Args:
        matrix (list): The input matrix.

    Returns:
        tuple: The shape of the matrix as a tuple of integers.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if len(matrix) > 0 else []
    return tuple(shape)
