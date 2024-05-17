#!/usr/bin/env python3
"""
This module defines a function to calculate the shape of a matrix.

"""


def matrix_shape(matrix):
    """

    Calculates the shape of a matrix.    
    Args:
        matrix (list): The matrix to calculate the shape of.

    Returns:
        list: A list of integers representing the shape of the matrix.

    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
