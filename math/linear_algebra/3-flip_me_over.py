#!/usr/bin/env python3
"""
This module defines a function to return the transpose of a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    Return the transpose of a 2D matrix.

    Args:
        matrix (list of lists): The matrix to be transposed.

    Returns:
        list of lists: The transposed matrix.
    """
    transposed = []
    for col in range(len(matrix[0])):
        new_row = []
        for row in range(len(matrix)):
            new_row.append(matrix[row][col])
        transposed.append(new_row)
    return transposed
