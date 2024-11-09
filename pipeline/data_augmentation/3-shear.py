#!/usr/bin/env python3
"""
Module for image augmentation techniques.
Contains functions for randomly shearing images.
"""

import tensorflow as tf


def shear_image(image, intensity):
    """
    Randomly shears an image.

    Args:
        image: A 3D tf.Tensor containing the image to shear
        intensity: The intensity with which the image should be sheared

    Returns:
        tf.Tensor: The sheared image
    """
    # Convert intensity to radians (as shear transform uses radians)
    shear = intensity * tf.random.uniform([]) * (3.14159 / 180.0)

    # Get image shape
    shape = tf.cast(tf.shape(image), dtype=tf.float32)
    height = shape[0]
    width = shape[1]

    # Create transformation matrix for shearing
    transform = [1.0, shear, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
    transform = tf.reshape(transform, [3, 3])

    # Convert to homogeneous coordinates
    coords = tf.meshgrid(
        tf.range(width),
        tf.range(height)
    )
    coords = tf.stack(coords, axis=-1)
    coords = tf.cast(coords, tf.float32)

    # Add batch dimension
    image = tf.expand_dims(image, 0)

    # Apply shear transformation
    sheared = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.cast(shape[:2], tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT"
    )

    # Remove batch dimension
    sheared = tf.squeeze(sheared, 0)

    # Ensure output has same dtype as input
    sheared = tf.cast(sheared, image.dtype)

    return sheared
