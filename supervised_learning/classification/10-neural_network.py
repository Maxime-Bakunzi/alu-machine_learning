#!/usr/bin/env python3
"""
This is a module of a neuron network.
"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing binary
    classification.
    """

    def __init__(self, nx, nodes):
        """Class constructor.

        Args:
            nx (int): The number of input features.
            nodes (int): The number of nodes in the hidden layer.

        Raises:
            TypeError: If nx is not an integer or nodes is not an integer.
            ValueError: If nx or nodes is less than 1.
        """

        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Private attributes of the hidden layer
        self.__W1 = np.random.randn(nodes, nx)  # Random normal initialization
        self.__b1 = np.zeros((nodes, 1))  # Bias initialization with zeros
        self.__A1 = 0  # Activated output of the output neuron initialize to 0

        # Private attributes of the output neuron
        self.__W2 = np.random.randn(1, nodes)  # Random normal initialization
        self.__b2 = 0  # Bias initialization to 0
        self.__A2 = 0  # Activated output of an output neuron initialized to 0

    @property
    def W1(self):
        """Getter of W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter of b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter of A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter of W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter of b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter of A2"""
        return self.__A2

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network.

        Args:
            x (numpy.ndarray): Input data of shape (nx, m)

        Returns:
            tuple: The private attributes __A1 and __A2, respectively.
        """

        # Calculates Z1 and A1 of the hidden layer
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(Z1)

        # Calculate Z2 and A2 of the output layer
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(Z2)

        return self.__A1, self.__A2
