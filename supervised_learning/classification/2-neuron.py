#!/usr/bin/env python3
"""
This is a module for creating for forward propagation.
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

    def forward_prop(self, X):
        """Calculates forward propagation using sigmoid activation function.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).

        Returns:
            numpy.ndarray: The activated output (A) of the neuron.
        """

        # Linear part: Z = W.X + b
        Z = np.matmul(self.__W, X) + self.__b

        # Sigmoid activation function: A = 1 / (1 + exp(-Z))
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A
