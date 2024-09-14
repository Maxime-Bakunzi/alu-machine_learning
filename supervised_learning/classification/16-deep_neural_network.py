#!/usr/bin/env python3
"""
This module about the deep neural network.
"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network preforming binary classification
    """

    def __init__(self, nx, layers):
        """
        class constructor
        Args:
            nx(int): number of input features
            layers (list): list representing the number of nodes in 
            each layer of the network
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")
        
        self.L = len(layers)  # number of layers
        self.cache = {}  # intermediary values of the network
        self.weights = {}  # weights and biases of the network

        # He et al. initialization of weights
        for l in range (1, self.L + 1):
            if l == 1:
                self.weights['W1'] = np.random.randn(layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W{}'.format(l)] = np.random.randn(layers[l - 1], layers[l - 2]) * np.sqrt(2 / layers[l - 2])
                self.weights['b{}'.format(l)] = np.zeros((layers[l - 1], 1))