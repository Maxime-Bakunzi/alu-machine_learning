#!/usr/bin/env python3
"""
Module containing the implementation of RMSProp optimization algorithm
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm.

    Args:
        loss: The loss of the network
        alpha: The learning rate
        beta2: The RMSProp weight
        epsilon: A small number to avoid division by zero

    Returns:
        The RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
