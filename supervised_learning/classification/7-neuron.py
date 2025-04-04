#!/usr/bin/env python3
"""
This is a module for training a neuron.
"""

import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """"Performs one pass of gradient on the neuron.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output of the neuron.
            alpha (float): Learning rate.

        Updates:
            __W: The weights vector.
            __b: The bias scalar.
        """

        m = Y.shape[1]
        dZ = A - Y  # Difference between predicted and actual labels
        dW = (1 / m) * np.dot(dZ, X.T)  # Gradient of the weights
        db = (1 / m) * np.sum(dZ)  # Gradient of the bias

        # update the weights and bias
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neauron

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m)
            iterations (int): Number of iterations to train over.
            alpha (float): Learning rate.
            verbose (bool): If True, prints cost information.
            graph (bool): If True, plots cost information.
            step (int): Frequency of printing and plotting.

        Raises:
            TypeError: If iterations or step is not an integer or
                       alpha is not a float .
            ValueError: If iterations, step or alpha are not positive
                        or step is not less than iterations.

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

        # Validate step
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []

        # Perform training over the specified number of iterations
        for i in range(iterations + 1):
            A = self.forward_prop(X)  # Forward propagation
            cost = self.cost(Y, A)  # Compute cost
            if i % step == 0 or i == iterations:
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                costs.append(cost)
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)  # Gradient descent

        # Plot graph if required
        if graph:
            plt.plot(np.arange(0, iterations + 1, step), costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        # Evaluate the training data after training
        return self.evaluate(X, Y)
