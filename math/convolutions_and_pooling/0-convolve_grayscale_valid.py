#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
    images (numpy.ndarray): Array of shape (m, h, w) containing
    multiple grayscale images.
    kernel (numpy.ndarray): Array of shape (kh, kw) containing
    the kernel for convolution.

    Returns:
    numpy.ndarray: The convolved images.
    """
    # Extract dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate dimensions of the output
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize the output array
    convolved = np.zeros(m, output_h, output_w)

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            convolved[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw]
                                        * kernel, axis=(1, 2))

    return convolved
