#!/usr/bin/env python3
"""
Module about the deep neural network performing binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Class constructor for DeepNeuralNetwork

        Args:
            nx (int): number of input features
            layers (list): list representing the number of nodes in each layer

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
            TypeError: If layers is not a list or is empty
            TypeError: If elements in layers are not all positive integers
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(
                map(lambda layer: isinstance(layer, int)
                    and layer > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)  # number of layers
        self.__cache = {}  # intermediary values of the network
        self.__weights = {}  # weights and bias of the network

        for le in range(1, self.__L + 1):
            layer_size = layers[le - 1]
            input_size = nx if le == 1 else layers[le - 2]

            self.__weights['W' + str(le)] = np.random.randn(
                layer_size, input_size) * np.sqrt(2 / input_size)
            self.__weights['b' + str(le)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter method of the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter method of the cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter method of the weights dictionary"""
        return self.__weights
