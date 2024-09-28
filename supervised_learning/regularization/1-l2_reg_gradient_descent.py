#!/usr/bin/env python3
"""
Module for L2 regularized gradient descent
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization

    Args:
        Y (numpy.ndarray): One-hot matrix of shape (classes, m) containing
        the correct labels
        weights (dict): Dictionary of the weights and biases of the neural
        network
        cache (dict): Dictionary of the outputs of each layer of the neural
        network
        alpha (float): The learning rate
        lambtha (float): The L2 regularization parameter
        L (int): The number of layers of the network

    Returns:
        None (weights and biases are updated in place)
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if l > 1:
            dZ = np.matmul(W.T, dZ) * (1 - np.square(A_prev))

        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db
