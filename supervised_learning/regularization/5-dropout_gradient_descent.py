#!/usr/bin/env python3
"""
Module for gradient descent with Dropout
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
     using gradient descent

    Args:
        Y (numpy.ndarray): One-hot matrix of shape (classes, m) containing
         the correct labels
        weights (dict): Dictionary of the weights and biases of the neural
         network
        cache (dict): Dictionary of the outputs and dropout masks of each
         layer
        alpha (float): Learning rate
        keep_prob (float): Probability that a node will be kept
        L (int): Number of layers of the network

    Returns:
        None (weights are updated in place)
    """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y

    for le in reversed(range(1, L + 1)):
        A_prev = cache['A{}'.format(le-1)]
        W = weights['W{}'.format(le)]
        b = weights['b{}'.format(le)]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if le > 1:
            dA_prev = np.dot(W.T, dZ)
            dA_prev *= cache['D{}'.format(le-1)]  # Apply dropout mask
            dA_prev /= keep_prob  # Scale the values
            dZ = dA_prev * (1 - np.square(A_prev))  # Derivative of tanh

        weights['W{}'.format(le)] -= alpha * dW
        weights['b{}'.format(le)] -= alpha * db
