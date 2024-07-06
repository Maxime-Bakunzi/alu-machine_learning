#!/usr/bin/env python3
"""
Convolution on images using multiple kernels.
"""

import numpy as np


def convolve(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
    - images: np.ndarray with shape (m, h, w, c) containing
    multiple grayscale images
    - kernel: np.ndarray with shape (kh, kw, c, nc) containing
    the kernel for convolution
    - padding: either a turple of (ph, pw), 'same', or
    'valid'
    - stride: a turple of (sh, sw)

    Returns:
    - numpy.ndarray containing the convolved images
    """

    # Extract dimensions
    m, h, w, c = images.shape
    kh, kw, c, nc = kernel.shape
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
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    # Calculate output dimensions
    out_h = (padded_h - kh) // sh + 1
    out_w = (padded_w - kw) // sw + 1

    # Initialize the output array
    convolved = np.zeros((m, out_h, out_w, nc))

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            for k in range(nc):
                slice_h_start = i * sh
                slice_h_end = slice_h_start + kh
                slice_w_start = j * sw
                slice_w_end = slice_w_start + kw
                convolved[:, i, j] = np.sum(padded_images
                                        [:, slice_h_start:slice_h_end,
                                         slice_w_start:slice_w_end, :]
                                        * kernel[:, :, :, k], axis=(1, 2, 3))

    return convolved
