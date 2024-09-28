#!/usr/bin/env python3
"""
Module for creating a TensorFlow layer with L2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer that includes L2 regularization

    Args:
        prev (tensor): A tensor containing the output of the previous layer
        n (int): The number of nodes the new layer should contain
        activation (callable): The activation function that should be used
         on the layer
        lambtha (float): The L2 regularization parameter

    Returns:
        tensor: The output of the new layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    return layer(prev)
