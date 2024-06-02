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

    integral_coeffs = [coef / (i + 1) for i, coef in enumerate(poly)]
    integral_coeffs.insert(0, C)  # Add integration constant

    # Round to integers if coefficient is very close to an integer
    integral_coeffs = [round(coef) if abs(coef - round(coef)) < 1e-10 else coef for coef in integral_coeffs]

    return integral_coeffs
