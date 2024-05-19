#!/usr/bin/env python3
"""
This module defines a function to perform matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Perform matrix multiplication.

    Args:
        mat1 (list of lists of ints/floats): The first matrix.
        mat2 (list of lists of ints/floats): The second matrix.

    Returns:
        list of lists: matrix that is the result of multiplying mat1 by mat2.
        None: If the matrices cannot be multiplied.
    """
    # Check if matrix multiplication is possible
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform the matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
