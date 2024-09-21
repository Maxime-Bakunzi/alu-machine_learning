#!/usr/bin/env python3
"""Module for creating the training operation for the network"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Create the training operation for the network.

    Args:
        loss (tf.Tensor): The loss of the network's prediction.
        alpha (float): The learning rate.

    Returns:
        tf.operation: An operation that trains the network using gradient
         descent.
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)

    return train_op
