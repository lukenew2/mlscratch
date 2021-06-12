"""Module containing classes for activation functions."""
import numpy as np

class Sigmoid():
    """
    Sigmoid activation function used in linear classification models.

        sigmoid(x) = 1 / (1 + exp(-x))

    where x is an array of scalar values.
    """
    def __call__(self, x):
        """Outputs a number between 0 and 1 used in binary classification."""
        return 1 / (1 + np.exp(-x))

class Softmax():
    """
    Softmax activation function used in linear classification models.

        softmax(x) = np.exp(x) / sum(np.exp(x))

    where x is an array of scaler values.

    Notes
    -----
    For numerical stability we subtract the largest value of each row in the
    array to avoid dividing by huge numbers in the exponentials.
    """
    def __call__(self, x):
        """
        Outputs a probability between 0 and 1 for each class in multiclass
        classification problems.
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 

        return e_x / np.sum(e_x, axis=1, keepdims=True)