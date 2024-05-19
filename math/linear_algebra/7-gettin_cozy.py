#!/usr/bin/env python3
"""
This module defines a function to concatenate two 2D matrices along an  axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two 2D matrices along a specific axis.

    Args:
        mat1 (list of lists of ints/floats): The first matrix.
        mat2 (list of lists of ints/floats): The second matrix.
        axis (int): The axis along to concatenate (0 for row, 1 for colum).

    Returns:
        list of lists: matrix that is the concatenation of mat1 and mat2.
        None: If the matrices cannot be concatenated.
    """
    # Check if concatenation is possible
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
