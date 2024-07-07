#!/usr/bin/env python3
"""
This module perfoms Pooling on images
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Perfoms pooling on images.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w, c) containing
    multiple images
    - kernal_sahpe: tuple of (kh, kw) containing the kernel shape for pooling
    - stride: turple of (sh, sw)
    - mode: indicates the type of pooling ('max' or 'avg')

    Returns:
    - numpy.ndarray containing the pooled images
    """

    # Extract the dimensions
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    # Initialize output array
    pooled_images = np.zeros((m, out_h, out_w, c))

    # Perform pooling
    for i in range(out_h):
        for j in range(out_w):
            # Extract the slice of the input image
            img_slice = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

            if mode == 'max':
                pooled_images[:, i, j, :] = np.max(img_slice, axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, i, j, :] = np.mean(img_slice, axis=(1, 2))

    return pooled_images
