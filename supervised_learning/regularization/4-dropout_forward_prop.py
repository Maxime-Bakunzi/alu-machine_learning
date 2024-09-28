#!/usr/bin/env python3
"""
Module for forward propagation using Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X (numpy.ndarray): Input data of shape (nx, m)
        weights (dict): Dictionary of the weights and biases of the neural
         network
        L (int): Number of layers in the network
        keep_prob (float): Probability that a node will be kept

    Returns:
        dict: Dictionary containing the outputs of each layer and the
         dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X

    for le in range(1, L + 1):
        W = weights['W{}'.format(le)]
        b = weights['b{}'.format(le)]
        A_prev = cache['A{}'.format(le-1)]
        Z = np.dot(W, A_prev) + b

        if le == L:
            # Softmax activation for the last layer
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            # Tanh activation for hidden layers
            A = np.tanh(Z)

            # Apply dropout
            D = np.random.rand(*A.shape) < keep_prob
            A *= D
            A /= keep_prob
            cache['D{}'.format(le)] = D

        cache['A{}'.format(le)] = A

    return cache
