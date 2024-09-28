#!/usr/bin/env python3
"""
Module containing the implementation of Adam optimization algorithm
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.

    Args:
        alpha: The learning rate
        beta1: The weight used for the first moment
        beta2: The weight used for the second moment
        epsilon: A small number to avoid division by zero
        var: A numpy.ndarray containing the variable to be updated
        grad: A numpy.ndarray containing the gradient of var
        v: The previous first moment of var
        s: The previous second moment of var
        t: The time step used for bias correction

    Returns:
        The updated variable, the new first moment, and the new second
         moment, respectively
    """
    # Update first moment
    v = beta1 * v + (1 - beta1) * grad

    # Update second moment
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Compute bias-corrected first moment
    v_corrected = v / (1 - beta1 ** t)

    # Compute bias-corrected second moment
    s_corrected = s / (1 - beta2 ** t)

    # Update variable
    var = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var, v, s
