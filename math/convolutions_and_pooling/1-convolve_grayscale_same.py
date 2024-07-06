#!/usr/bin/env python3
"""
Same convolution on a grayscale images
"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on a grayscale images.

    Args:
    images (numpy.ndarray): Array of shape (m, h, w) containing multiple
    grayscale images.
    kernel (numpy.ndarray): Array of shape (kh, kw) containing kernel
    for the convolution.

    Returns:
    numpy.ndarray: The convolved images
    """

    # Extract dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the padding
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad the images
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h),
                                    (pad_w, pad_w)), mode='constant')

    # Initialize the output array
    convolved = np.zeros((m, h, w))

    # Perform convolution
    for i in range(h):
        for j in range(w):
            convolved[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw]
                                        * kernel, axis=(1, 2))

    return convolved
