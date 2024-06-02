#!/usr/bin/env python3
"""
This module provides a function to calculate the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial represented by
    a list of coefficients.

    Args:
        poly (list): List of coefficients representing a polynomial.

    Returns:
        list: New list of coefficients representing the
        derivative of the polynomial, or None if the input is not valid.
    """
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float))
       for coef in poly):
        return None

    if len(poly) == 0:
        return None

    derivative = [i * poly[i] for i in range(1, len(poly))]

    return derivative if derivative else [0]
