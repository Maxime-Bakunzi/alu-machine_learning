#!/usr/bin/env python3
"""
This is a module for creating a neuron with Private instance attributes
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
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Private attributes
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector."""
        return self.__W

    @property
    def b(self):
        """Getter for thr bias."""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output."""
        return self.__A
