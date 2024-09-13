#!/usr/bin/env python3
"""
This is a module for a neuron network.
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

        # Weights and bias for the hidden layer
        self.W1 = np.random.randn(nodes, nx)  # Random normal initialization
        self.b1 = np.zeros((nodes, 1))  # Bias initialization with zeros
        self.A1 = 0  # Activated output of the output neuron initialize to 0

        # Weights and biases for the output neuron
        self.W2 = np.random.randn(1, nodes)  # Random normal initialization
        self.b2 = 0  # Bias initialization to 0
        self.A2 = 0  # Activated output of the output neuron initialized to 0
