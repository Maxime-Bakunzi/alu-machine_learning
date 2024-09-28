#!/usr/bin/env python3
"""Module for calculating the weighted moving average of a data set"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    Args:
    data (list): The list of data to calculate the moving average of
    beta (float): The weight used for the moving average

    Returns:
    list: A list containing the moving averages of data
    """
    v = 0
    moving_averages = []
    for i, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        # Bias correction
        moving_averages.append(v / (1 - beta ** (i + 1)))
    return moving_averages
