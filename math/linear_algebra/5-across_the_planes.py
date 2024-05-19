#!/usr/bin/env python3
"""
This module defines a function to add two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Add two 2D matrices element-wise.

    Args:
        mat1 (list of lists of ints/floats): The first matrix.
        mat2 (list of lists of ints/floats): The second matrix.

    Returns:
        list of lists of ints/floats: A new matrix with the element-wise sums.
        None: If the matrices are not the same shape.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)

    return result
