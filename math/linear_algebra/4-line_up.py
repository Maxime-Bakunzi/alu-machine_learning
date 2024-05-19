#!/usr/bin/env python3
"""
This module defines a function to add two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Add two arrays element-wise.

    Args:
        arr1 (list of ints/floats): The first array.
        arr2 (list of ints/floats): The second array.

    Returns:
        list of ints/floats: A new list with the element-wise sums.
        None: If the arrays are not the same shape.
    """
    if len(arr1) != len(arr2):
        return None

    return [arr1[i] + arr2[i] for i in range(len(arr1))]
