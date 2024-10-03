#!/usr/bin/env python3
"""Module for calculating specificity from a confusion matrix"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion numpy.ndarray of shape
                                   (classes, classes) where row indices
                                   represent the correct labels and column
                                   indices represent the predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing the specifity
                        of each class.
    """
    true_negatives = np.sum(confusion) - np.sum(confusion, axis=0) - np.sum(
        confusion, axis=1) + np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - np.diag(confusion)
    return true_negatives / (true_negatives + false_positives)
