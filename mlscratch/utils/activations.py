"""Module containing classes for activation functions."""
import numpy as np

class Sigmoid():
    """
    Sigmoid activation function used in linear classification models.

        1 / (1 - exp(x))

    where x is an array of scalar values.
    """
    def __call__(self, x):
        """Outputs a number between 0 and 1 used in binary classification."""
        return 1 / (1 + np.exp(-x))