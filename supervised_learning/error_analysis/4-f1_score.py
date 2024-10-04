#!/usr/bin/env python3
"""Module for calculating F1 score from a confusion matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion numpy.ndarray of shape
                                    (classes, classes) where row indices
                                    represent the correct labels and column
                                    indices represent the predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing the F1 score of
                        each class
    """
    prec = precision(confusion)
    sens = sensitivity(confusion)
    return 2 * (prec * sens) / (prec + sens)
