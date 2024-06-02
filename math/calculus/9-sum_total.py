#!/usr/bin/env python3
"""
This module provides a function to calculate the sum of squares
of the first n natural numbers.
"""


def summation_i_squared(n):
    """
    Calculates the sum of squares of integers from 1 to n.

    Args:
        n (int): The stopping condition.

    Returns:
        int: The sum of squares if n is a valid number, otherwise None.
    """
    if not isinstance(n, int) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6
