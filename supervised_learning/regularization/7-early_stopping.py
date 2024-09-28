#!/usr/bin/env python3
"""Module for early stopping implementation in gradient descent."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should stop early.

    Args:
        cost (float): Current validation cost of the neural network.
        opt_cost (float): Lowest recorded validation cost of the neural
         network.
        threshold (float): Threshold used for early stopping.
        patience (int): Patience count used for early stopping.
        count (int): Count of how long the threshold has not been met.

    Returns:
        tuple: (boolean, int)
            - boolean: Whether the network should be stopped early.
            - int: The updated count.
    """
    if opt_cost - cost > threshold:
        return False, 0
    else:
        if count + 1 >= patience:
            return True, count + 1
        else:
            return False, count + 1
