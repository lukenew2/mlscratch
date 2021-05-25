"""Module containing functions to compute common machine learning metrics."""
import numpy as np 

def mean_squared_error(y_true, y_pred, squared=True):
    """
    Mean squared error regression loss function.

    Parameters 
    ----------
    y : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    squared : bool, default=True
        If True returns MSE, if False returns RMSE.

    Returns 
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0).
    """
    # Make sure inputs are numpy arrays.
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Calculate array of errors for each instance.
    errors = np.average((y_true - y_pred) ** 2, axis=0)

    # Calculates square root of each error if squared=False.
    if not squared:
        errors = np.sqrt(errors)

    # Return average error across all instances.
    return np.average(errors)

