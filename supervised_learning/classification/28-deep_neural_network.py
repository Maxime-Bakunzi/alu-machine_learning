#!/usr/bin/env python3
"""Module that defines a deep neural network with binary classification"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Class constructor for DeepNeuralNetwork

        Args:
            nx (int): number of input features
            layers (list): list representing the number of nodes in each layer
            activation (str): 'sig' for sigmoid, 'tanh' for tanh activation
        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
            TypeError: If layers is not a list or is empty
            TypeError: If elements in layers are not all positive integers
            ValueError: If activation is neither 'sig' nor 'tanh'
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
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for le in range(1, self.L + 1):
            layer_size = layers[le - 1]
            input_size = nx if le == 1 else layers[le - 2]

            self.__weights['W' + str(le)] = np.random.randn(
                layer_size, input_size) * np.sqrt(2 / input_size)
            self.__weights['b' + str(le)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter of the cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter of the weights"""
        return self.__weights

    @property
    def activation(self):
        """Getter of the activation function"""
        return self.__activation

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def tanh(self, Z):
        """Tanh activation function"""
        return np.tanh(Z)

    def softmax(self, Z):
        """Softmax activation function"""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_prop(self, X):
        """Calculates the propagation of the neural network

        Args:
            X (array): is a numpy.ndarray with shape (nx, m) that
            contains the input data
        """

        self.__cache['A0'] = X

        for le in range(1, self.__L + 1):
            W = self.__weights['W' + str(le)]
            b = self.__weights['b' + str(le)]
            A_prev = self.__cache['A' + str(le - 1)]

            Z = np.dot(W, A_prev) + b

            if le == self.__L:
                A = self.softmax(Z)
            else:
                if self.__activation == 'sig':
                    A = self.sigmoid(Z)
                else:
                    A = self.tanh(Z)

            self.__cache['A' + str(le)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Correct labels with shape(1, m)
            A (numpy.ndarray): Activated output with shape (1, m)

        Returns:
            The cost (logistic regression cost)
        """

        m = Y.shape[1]
        epsilon = 1e-15
        cost = -(1 / m) * np.sum(Y * np.log(A + epsilon))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's prediction

        Args:
            X (numpy.ndarray): input data with shape (nx, m)
            Y (numpy.ndarray): Correct labels with shape (1, m)

        Returns:
            The neuron's prediction, The cost of the network
        """

        A, _ = self.forward_prop(X)
        prediction = np.argmax(A, axis=0)
        prediction_one_hot = np.eye(Y.shape[0])[prediction].T

        cost = self.cost(Y, A)

        return prediction_one_hot, cost

    def sigmoid_derivative(self, A):
        """Derivative of the sigmoid function for backpropagation"""
        return A * (1 - A)

    def tanh_derivative(self, A):
        """Derivative of the tanh function for backpropagation"""
        return 1 - A ** 2

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y (numpy.ndarray): correct labels with shape (1, m)
            cache (dictionary): intermediary values of the network
            alpha (float): learning rate
        """

        m = Y.shape[1]
        dZ = cache['A' + str(self.__L)] - Y

        for layer in range(self.__L, 0, -1):
            A_prev = cache['A' + str(layer - 1)]

            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if layer > 1:
                W = self.__weights['W' + str(layer)]
                dA = np.dot(W.T, dZ)
                A = cache['A' + str(layer - 1)]
                if self.__activation == 'sig':
                    dZ = dA * self.sigmoid_derivative(A)
                else:
                    dZ = dA * self.tanh_derivative(A)

            self.__weights['W' + str(layer)] -= alpha * dW
            self.__weights['b' + str(layer)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m)
            iterations (int): Number of iterations to train over.
            alpha (float): Learning rate.

        Returns:
            tuple: The evaluation of the training data after iterations
            of training
        """

        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("alpha must be a positive float")

        costs = []

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            if verbose and i % step == 0 or i == iterations - 1:
                cost = self.cost(Y, A)
                costs.append(cost)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        if graph:
            plt.plot(range(0, iterations, step), costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
