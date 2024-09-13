#!/usr/bin/env python3
"""
This is a module for creating a neuron where we calculate the cost
using logistic regression.
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

    def cost(self, Y, A):
        """Calculates the cost using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels (1, m) for the input data.
            A (numpy.ndarray): Activated output (1, m) for each example.

        Returns:
            float: The cost of the model.

        """
        m = Y.shape[1]  # number of examples
        # Compute the cost using cross-entropy
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) *
                                  np.log(1.0000001 - A))

        return cost

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).

        Returns:
            numpy.ndarray: Predicted labels (1 or 0) with shape (1, m).
            float: Cost of the network.
        """

        # Perform forward propagation to get A
        A = self.forward_prop(X)

        # Make prediction: 1 if A >= 0.5, otherwise 0
        predictions = np.where(A >= 0.5, 1, 0)

        # Calculate the cost
        cost = self.cost(Y, A)

        return predictions, cost
