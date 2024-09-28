#!/usr/bin/env python3
"""
Module for L2 regularization cost calculation using TensorFlow
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost (tensor): A tensor containing the cost of the network without
         L2 regularization

    Returns:
        tensor: A tensor containing the cost of the network accounting for
         L2 regularization
    """
    l2_cost = tf.losses.get_regularization_losses()
    return cost + l2_cost
