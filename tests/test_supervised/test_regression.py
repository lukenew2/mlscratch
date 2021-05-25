"""Module containing tests for mlscratch/supervised/regression.py"""
import numpy as np
from numpy.core.fromnumeric import mean
from numpy.testing import assert_array_almost_equal
import pytest

from mlscratch.utils.preprocessing import StandardScaler
from mlscratch.utils.preprocessing import PolynomialFeatures
from mlscratch.supervised.regression import LinearRegression


def test_linear_regression_simple(simple_data):
    """Test LinearRegression with ordinary least squares and batch gd."""
    X = [[1], [2]]
    y = [[1], [2]]

    for solver in ['bgd', 'lstsq']:
        reg = LinearRegression(solver='lstsq')
        reg.fit(X, y)

        assert_array_almost_equal(reg.coef_, [[0.], [1.]])
        assert_array_almost_equal(reg.predict(X), [[1.], [2.]])

def test_linear_regression_complex():
    """Test LinearRegression using batch gradient descent."""
    X = np.random.rand(100, 1)
    y = 5 * X + 3

    for solver in ['bgd', 'lstsq']:
        reg = LinearRegression(n_iter=10000, solver=solver)
        reg.fit(X, y)

        assert_array_almost_equal(reg.coef_, [[3.], [5.]])
        assert_array_almost_equal(reg.predict(X), y)

def test_polynomial_regression():
    """
    Test LinearRegression using PolynomialFeatures and StandardScaler classes.
    """
    X = 6 * np.random.rand(100, 1) - 3
    y = 0.5 * X ** 2 + X + 2

    poly = PolynomialFeatures(2)

    poly_features = poly.fit_transform(X)

    scaler = StandardScaler()
    scaler.fit(poly_features)

    poly_features_scaled = scaler.transform(poly_features)

    for solver in ['bgd', 'lstsq']:
        reg = LinearRegression(n_iter = 10000, solver=solver)
        reg.fit(poly_features_scaled, y)

        assert_array_almost_equal(reg.predict(poly_features_scaled), y)


