#!/usr/bin/env python3
"""
This module defines a function that  concatenate matrices.

"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two matrices along a specific axis.

    Args:
        mat1 (numpy.ndarray): First input matrix.
        mat2 (numpy.ndarray): Second input matrix.
        axis (int, optional): Axis along which to concatenate. Defaults to 0.

    Returns:
        numpy.ndarray: Concatenated matrix.
    """
    return np.concatenate((mat1, mat2), axis=axis)
