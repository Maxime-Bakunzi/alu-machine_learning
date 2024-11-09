#!/usr/bin/env python3
"""
Module for image augmentation techniques.
Contains functions for randomly changing image brightness.
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image: A 3D tf.Tensor containing the image to change
        max_delta: The maximum amount the image should be brightened (or darkened)

    Returns:
        tf.Tensor: The brightness-adjusted image
    """
    return tf.image.random_brightness(image, max_delta)
