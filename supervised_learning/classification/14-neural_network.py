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

    def cost(self, Y, A):
        """Calculates the cost using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels (1, m) of the input data.
            A (numpy.ndarray): Activated output (1, m) of each example.

        Returns:
            float: The cost of the model.

        """
        m = Y.shape[1]  # number of examples
        # Compute the cost using cross-entropy
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) *
                                  np.log(1.0000001 - A))

        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.

        Returns:
            tuple: Predicted labels and the cost.
        """

        # perform forward propagation
        self.forward_prop(X)

        # Calculate the predictions
        A2 = self.__A2
        prediction = np.where(A2 >= 0.5, 1, 0)

        # Calculates the cost
        cost = self.cost(Y, A2)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """"Performs one pass of gradient on the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A1 (numpy.ndarray): Output of the hidden layer.
            A2 (numpy.ndarray): Predicted output.
            alpha (float): Learning rate.
        """

        m = X.shape[1]

        # Calculate the gradient of the output layer
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Calculate the gradient of the hidden layer
        dZ1 = np.dot(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # update the weights and bias
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

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
            A1, A2 = self.forward_prop(X)  # Forward propagation
            self.gradient_descent(X, Y, A1, A2, alpha)  # Gradient descent

        # Evaluate the training data after training
        return self.evaluate(X, Y)
