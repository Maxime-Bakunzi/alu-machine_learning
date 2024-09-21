#!/usr/bin/env python3
"""Module for creating a layer in TensorFlow"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Create a layer for a neural network.

    Args:
        prev (tf.Tensor): The tensor output of the previous layer.
        n (int): The number of nodes in the layer to create.
        activation (function): The activation function that layer should use.

    Returns:
        tf.Tensor: The tensor output of the layer.
    """

    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer, name='layer')

    return layer(prev)
