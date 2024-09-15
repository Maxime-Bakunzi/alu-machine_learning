#!/usr/bin/env python3
"""Module that defines a deep neural network
with binary classification"""

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
            activation (str): activation function to use in hidden layers

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
            TypeError: If layers is not a list or is empty
            TypeError: If elements in layers are not all positive integers
            ValueError: If activation is not 'sig' or 'tanh'
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for l in range(1, self.__L + 1):
            self.__weights['W' + str(l)] = np.random.randn(layers[l-1], nx if l ==
                                                           1 else layers[l-2]) * np.sqrt(2 / (nx if l == 1 else layers[l-2]))
            self.__weights['b' + str(l)] = np.zeros((layers[l-1], 1))

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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)

        Returns:
            tuple: Output of the neural network and cache
        """
        self.__cache['A0'] = X
        for l in range(1, self.__L + 1):
            Z = np.dot(
                self.__weights['W' + str(l)], self.__cache['A' + str(l-1)]) + self.__weights['b' + str(l)]
            if l == self.__L:
                self.__cache['A' + str(l)] = self.softmax(Z)
            else:
                if self.__activation == 'sig':
                    self.__cache['A' + str(l)] = 1 / (1 + np.exp(-Z))
                else:  # tanh
                    self.__cache['A' + str(l)] = np.tanh(Z)
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A (numpy.ndarray): Activated output with shape (1, m)

        Returns:
            float: The cost
        """
        m = Y.shape[1]
        epsilon = 1e-15
        return -1/m * np.sum(Y * np.log(A + epsilon))

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
            Y (numpy.ndarray): Correct labels with shape (1, m)

        Returns:
            tuple: Prediction (numpy.ndarray) and cost (float)
        """
        A, _ = self.forward_prop(X)
        prediction = np.eye(Y.shape[0])[np.argmax(A, axis=0)].T
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m)
            cache (dict): Intermediary values of the network
            alpha (float): Learning rate
        """
        m = Y.shape[1]
        dZ = cache['A' + str(self.__L)] - Y
        for l in reversed(range(1, self.__L + 1)):
            A_prev = cache['A' + str(l-1)]
            dW = (1/m) * np.dot(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                if self.__activation == 'sig':
                    dZ = np.dot(
                        self.__weights['W' + str(l)].T, dZ) * (A_prev * (1 - A_prev))
                else:  # tanh
                    dZ = np.dot(
                        self.__weights['W' + str(l)].T, dZ) * (1 - A_prev ** 2)
            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
            Y (numpy.ndarray): Correct labels with shape (1, m)
            iterations (int): Number of iterations to train over
            alpha (float): Learning rate
            verbose (bool): Whether to print information about the training
            graph (bool): Whether to graph information about the training
            step (int): Step for printing verbose information and graphing

        Returns:
            tuple: Evaluation of the training data after iterations of training
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost = self.cost(Y, A)

            if verbose and (i % step == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))
            if graph and (i % step == 0 or i == iterations):
                costs.append(cost)

        if graph:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        Args:
            filename (str): The file to which the object should be saved
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        Args:
            filename (str): The file from which the object should be loaded

        Returns:
            DeepNeuralNetwork: The loaded object, or None if filename doesn't exist
        """
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None

    def softmax(self, Z):
        """Compute softmax activation function"""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
