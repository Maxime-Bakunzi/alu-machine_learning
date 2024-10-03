#!/usr/bin/env python3
"""Module for calculating sensitivity from a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion numpy.ndarray of shape
                                    (classes, classes) where row indices
                                    representing the correct labels and column
                                    indices represent the predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing the sensitivity
                        of each class.
    """
    true_positives = np.diag(confusion)
    class_totals = np.sum(confusion, axis=1)
    return true_positives / class_totals
