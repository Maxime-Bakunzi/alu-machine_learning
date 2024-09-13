#!/usr/bin/env python3
"""
This is a module for creating a neuron
"""

import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Class constructor.

        Args:
            nx (int): The number of input feature to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        # input validation
        if not isinstance(nx, int):
            raise TypeError("nx must be integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # weights, bias, and activated output initialization
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
