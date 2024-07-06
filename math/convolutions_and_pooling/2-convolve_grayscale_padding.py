#!/usr/bin/env python3
"""
Convolution on grayscale images with custom padding
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with a custom padding.

    Args:
    images (numpy.ndarray): Array of shape (m, h, w) containing
    multiple grayscale images.
    kernel (numpy.ndarray): Array of shape (kh, kw) containing
    tghe kernel of convolution.
    padding (turple): Tuple of (ph, pw) for the height and width padding.

    Returns:
    numpy.ndarray: The convolved images.
    """

    # Extract dimansions
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad the images
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    # Calculate new dimensions
    new_h = h + 2 * ph - kh + 1
    new_w = w + 2 * pw - kw + 1

    # Initialize the output array
    convolved = np.zeros((m, new_h, new_w))

    # Perform convolution
    for i in range(new_h):
        for j in range(new_w):
            convolved[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw]
                                        * kernel, axis=(1, 2))

    return convolved
