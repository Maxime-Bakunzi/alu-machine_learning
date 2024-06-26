#!/usr/bin/env python3
"""
This module calculates the intersection of obtaining
 this data with the various hypothetical probabilities
"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data given various hypothetical
    probabilities of developing severe side effects.

    Parameters:
    x (int): The number of patients that develop severe side effects.
    n (int): The total number of patients observed.
    P (numpy.ndarray): 1D array containing various hypothetical probabilities
                       of developing severe side effects.

    Returns:
    numpy.ndarray: 1D array containing the likelihood of obtaining
                   the data, x and n, for each probability in P, respectively.
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is " +
                         "greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate likelihood using the binomial distribution formula
    factorial = np.math.factorial
    binom_coeff = factorial(n) / (factorial(x) * factorial(n - x))
    likelihoods = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining this data
    with the various hypothetical probabilities.

    Parameters:
    x (int): The number of patients that develop severe side effects.
    n (int): The total number of patients observed.
    P (numpy.ndarray): 1D array containing the various hypothetical
                       probabilities of developing severe side effects.
    Pr (numpy.ndarray): 1D array containing the prior beliefs of P.

    Returns:
    numpy.ndarray: 1D array containing the intersection of obtaining
                   x and n with each probability in P, respectively.
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is " +
                         "greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate likelihood
    L = likelihood(x, n, P)

    # Calculate intersection
    intersection_values = L * Pr

    return intersection_values
