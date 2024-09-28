#!/usr/bin/env python3
"""Module for updating variables using momentum optimization"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization
     algorithm.

    Args:
    alpha (float): The learning rate
    beta1 (float): The momentum weight
    var (numpy.ndarray): The variable to be updated
    grad (numpy.ndarray): The gradient of var
    v (numpy.ndarray): The previous first moment of var

    Returns:
    tuple: The updated variable and the new moment, respectively
    """
    # Calculate the momentum
    v = beta1 * v + (1 - beta1) * grad

    # Update the variable
    var = var - alpha * v

    return var, v
