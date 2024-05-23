#!/usr/bin/env python3
"""
This module provides functions to calculate the determinant and
the minor matrix of a given matrix.
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Args:
        matrix (list of lists): A square matrix.

    Returns:
        int or float: The determinant of the matrix.
    """
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(len(matrix)):
        minor = [row[:c] + row[c+1:] for row in matrix[1:]]
        cofactor = (-1) ** c * matrix[0][c]
        det += cofactor * determinant(minor)
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a given matrix.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Raises:
        TypeError: If the input is not a list of lists.
        ValueError: If the matrix is not a non-empty square matrix.

    Returns:
        list of lists: The minor matrix of the input matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix[i])):
            submatrix = [row[:j] + row[j+1:] for row in (matrix[:i] +
                                                        matrix[i+1:])]
            minor_row.append(determinant(submatrix))
        minor_matrix.append(minor_row)

    return minor_matrix
