#!/usr/bin/env python3
"""
This module provides a function to calculate the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial.

    Args:
        poly (list): List of coefficients representing a polynomial.
        C (int): Integration constant (default is 0).

    Returns:
        list: New list of coefficients representing
        the integral of the polynomial, or None if the input is not valid.
    """
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float))
       for coef in poly):
        return None
    if not isinstance(C, int):
        return None

    
    if poly==[] or poly == [0]:
        return [C]

    integral_coeffs = [coef / (i + 1) for i, coef in enumerate(poly)]
    integral_coeffs.insert(0, C)  # Add integration constant

    # Round to integers if coefficient is very close to an integer
    integral_coeffs = [int(coef) if isinstance(coef, float) and coef.is_integer() else coef for coef in integral_coeffs]

    # Remove unnecessary trailing zeros (not including the constant term)
    while len(integral_coeffs) > 1 and integral_coeffs[-1] == 0:
        integral_coeffs.pop()

    return integral_coeffs
