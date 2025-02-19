#!/usr/bin/env python3
"""
Neural Style Transfer Implementation
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class that performs tasks for neural style transfer
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize the NST class
        Args:
            style_image: image used as style reference (numpy.ndarray)
            content_image: image used as content reference (numpy.ndarray)
            alpha: weight for content cost
            beta: weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or len(style_image.shape) != 3 \
                or style_image.shape[2] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')

        if not isinstance(content_image, np.ndarray) or len(content_image.shape) != 3 \
                or content_image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        Args:
            image: numpy.ndarray of shape (h, w, 3) containing the image
        Returns:
            scaled image as a tf.tensor
        """
        if not isinstance(image, np.ndarray) or len(image.shape) != 3 \
                or image.shape[2] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')

        h, w, _ = image.shape

        # Determine new dimensions while maintaining aspect ratio
        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)

        # Resize image using bicubic interpolation
        image = tf.image.resize_bicubic(
            tf.expand_dims(image, 0), (h_new, w_new))

        # Scale pixel values to range [0, 1]
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)

        return image
