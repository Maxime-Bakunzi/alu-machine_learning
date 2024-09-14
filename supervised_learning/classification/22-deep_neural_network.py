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

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): input data with shape (nx ,m)

        Returns:
            The output of the neural and the cache respectively
        """

        self.__cache['A0'] = X  # Input data is A0

        for l in range(1, self.__L + 1):
            Wl = self.__weights['W{}'.format(l)]
            bl = self.__weights['b{}'.format(l)]
            Al_prev = self.__cache['A{}'.format(l - 1)]
            Zl = np.dot(Wl, Al_prev) + bl  # Linear transformation
            Al = self.sigmoid(Zl)  # Apply sigmoid activation function
            self.__cache['A{}'.format(l)] = Al  # Actvated output in cache

        return Al, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Correct labels with shape(1, m)
            A (numpy.ndarray): Activated output with shape (1, m)

        Returns:
            The cost (logistic regression cost)
        """

        m = Y.shape[1]  # number of examples

        # Compute the cost using the logistic regression cost function
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

        return cost

    def evaluate(self, X, Y):
        """
        Evaluate teh neural network's prediction

        Args:
            X (numy.ndarray): input data with shape (nx, m)
            Y (numpy.ndarray): Corrext labels with shape (1, m)

        Returns:
            The neuron's prediction
            The cost of the network
        """

        # Forward propagation to get predictions
        A, _ = self.forward_prop(X)

        # Predict based om A, with a threshold of 0.5
        prediction = np.where(A >= 0.5, 1, 0)

        # Compute the cost
        cost = self.cost(Y, A)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient on the neural network

        Args:
            Y (numpy.ndarray): correct labels with shape (1, m)
            cache (dictionary): intermediary values of the network
            alpha (float) : learning rate
        """

        m = Y.shape[1]
        L = self.__L

        # Backpropagation
        dZ = cache['A' + str(L)] - Y
        for l in reversed(range(1, L + 1)):
            A_prev = cache['A' + str(l - 1)]
            W = self.__weights['W' + str(l)]
            b = self.__weights['b' + str(l)]

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if l > 1:
                dA = np.dot(W.T, dZ)
                dZ = dA * (A_prev * (1 - A_prev))

            # Update weights and biases
            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m)
            iterations (int): Number of iterations to train over.
            alpha (float): Learning rate.

        Raises:
            TypeError: If iterations is not integer or alpha is not float.
            ValueError: If iterations or alpha are not positive.

        Returns:
            tuple: The evaluation of the trainiing data after
            iterations of training
        """

        # Validate iterations
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        # Validate alpha
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Perform training over the specified number of iterations
        for i in range(iterations):
            A, cache = self.forward_prop(X)  # Forward propagation

            # Compute cost evry 100 iterations of monitoring
            if i % 100 == 0:
                cost = self.cost(Y, A)
                print("Cost after {} iterations: {}".format(i, cost))

            # Perform one step of gradient descent
            self.gradient_descent(Y, cache, alpha)

        # Evaluate the training data after training
        return self.evaluate(X, Y)
