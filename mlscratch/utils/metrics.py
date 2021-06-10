"""Module containing functions to compute common machine learning metrics."""
import numpy as np 

def accuracy_score(y_true, y_pred):
    """
    Classification performance metric that computes the accuracy of y_true
    and y_pred.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth correct labels.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    C : float
        Accuracy score.
    """
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    accuracy = correct / len(y_true)

    return accuracy 

def mean_squared_error(y_true, y_pred, squared=True):
    """
    Mean squared error regression loss function.

    Parameters 
    ----------
    y_true : array-like of shape (n_samples,)
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

def r2_score(y_true, y_pred):
    """
    R^2 regression score function.

        R^2 = 1 - SS_res / SS_tot

    where SS_res is the residual sum of squares and SS_tot is the total
    sum of squares.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    score : float
        R^2 score.
    """
    # Residual sum of squares.
    numerator = ((y_true - y_pred) ** 2).sum(axis=0)

    # Total sum of squares.
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0)

    # R^2.
    score = 1 - numerator / denominator

    return score 