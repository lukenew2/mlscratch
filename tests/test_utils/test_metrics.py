"""Module containing tests for mlscratch/utils/metrics.py"""
import numpy as np
from numpy.core.fromnumeric import mean
from numpy.testing import assert_almost_equal
import pytest 

from mlscratch.utils.metrics import mean_squared_error 

def test_mean_squared_error(n_samples=50):
    y_true = np.arange(n_samples)
    y_pred = y_true + 1

    assert_almost_equal(mean_squared_error(y_true, y_pred), 1.)

def test_mean_squared_error_squared():
    # Calculate MSE.
    mse1 = mean_squared_error([[1]], [[10]], squared=True)
    # Calculate RMSE.
    mse2 = mean_squared_error([[1]], [[10]], squared=False)
    # Assert sqrt(MSE) = RMSE
    assert np.sqrt(mse1) == pytest.approx(mse2)

 
