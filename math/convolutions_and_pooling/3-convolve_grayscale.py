#!/usr/bin/env python3
"""
Convolution on grayscale images
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
    images: np.ndarray with shape (m, h, w) containing
    multiple grayscale images
    kernel: np.ndarray with shape (kh, kw) containing
    the kernel for convolution
    padding: either a turple of (ph, pw), 'same', or
    'valid'
    stride: a turple of (sh, sw)

    Returns:
    np.ndarray containing the convolved images
    """

    # Extract dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine the padding values
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = (0, 0)
    else:
        ph, pw = padding

    # Calculate the padding
    padded_h = h + 2 * ph
    padded_w = w + 2 * pw

    # Pad the image
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    # Calculate new dimension
    new_h = (padded_h - kh) // sh + 1
    new_w = (padded_w - kw) // sw + 1

    # Initialize the output array
    convolved = np.zeros((m, new_h, new_w))

    # Perform convolution
    for i in range(new_h):
        for j in range(new_w):
            slice_h_start = i * sh
            slice_h_end = slice_h_start + kh
            slice_w_start = j * sw
            slice_w_end = slice_w_start + kw
            convolved[:, i, j] = np.sum(padded_images
                                        [:, slice_h_start:slice_h_end,
                                         slice_w_start:slice_w_end]
                                        * kernel, axis=(1, 2))

    return convolved
