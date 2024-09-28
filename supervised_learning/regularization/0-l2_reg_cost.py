#!/usr/bin/env python3
"""
Module for L2 regularization cost calculation
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost (float): The cost of the network without L2 regularization
        lambtha (float): The regularization parameter
        weights (dict): A dictionary of the weights and biases of the neural
         network
        L (int): The number of layers in the neural network
        m (int): The number of data points used

    Returns:
        float: The cost of the network accounting for L2 regularization
    """
    l2_cost = 0
    for l in range(1, L + 1):
        l2_cost += np.sum(np.square(weights['W{}'.format(l)]))

    l2_cost = (lambtha / (2 * m)) * l2_cost
    return cost + l2_cost
