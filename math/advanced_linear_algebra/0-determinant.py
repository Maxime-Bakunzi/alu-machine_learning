#!/usr/bin/env python3
"""
This module provides a function to calculate the determinant of a matrix.
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Args:
        matrix (list of lists): The matrix whose determinant is to be
        calculated.

    Raises:
        TypeError: If the input is not a list of lists.
        ValueError: If the matrix is not square.

    Returns:
        int or float: The determinant of the matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    if len(matrix) == 0 or not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case for a 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]

    # Recursive case for NxN matrix
    def minor(matrix, i, j):
        """
        Calculates the minor of matrix excluding the i-th row and j-th column.

        Args:
            matrix (list of lists): The matrix.
            i (int): The row to exclude.
            j (int): The column to exclude.

        Returns:
            list of lists: The minor matrix.
        """
        return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

    n = len(matrix)
    det = 0
    for col in range(n):
        submatrix = minor(matrix, 0, col)
        cofactor = ((-1) ** col) * matrix[0][col]
        det += cofactor * determinant(submatrix)

    return det
