#!/usr/bin/env python3
"""Module for creating a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Create a confusion matrix.

    Args:
        labels (numpy.ndarray): One-hot array of shape (m, classes) containing
                                the correct labels for each data point.
        logits (numpy.ndarray): One-hot array of shape (m, classes) containing
                                the predicted labels.

    Returns:
        numpy.ndarray: Confusion matrix of shape (classes, classes) with row
                        indices representing the corrct labels and column
                        indices representing the predicted labels.
    """
    return np.matmul(labels.T, logits)
