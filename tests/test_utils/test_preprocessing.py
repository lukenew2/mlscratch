"""Module containing tests for mlscratch/utils/preprocessing.py."""

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from mlscratch.utils.preprocessing import PolynomialFeatures
from mlscratch.utils.preprocessing import StandardScaler

def test_standard_scaler():
    """Simple test for standard scaler."""
    X = [[0, 0], [0, 0], [1, 1], [1, 1]]

    scaler = StandardScaler()

    scaler.fit(X)

    assert_array_almost_equal(scaler.mean_, [0.5, 0.5])
    assert_array_almost_equal(scaler.var_, [0.25, 0.25])

    X_std = scaler.transform(X)
    assert_array_almost_equal(X_std, [[-1., - 1.],
                                      [-1., - 1.],
                                      [ 1.,   1.],
                                      [ 1.,   1.]])

    assert_array_almost_equal(scaler.transform([[2, 2]]), [[3., 3.]])

    X_inv = scaler.inverse_transform(X_std)

    assert_array_almost_equal(X_inv, X)

def test_polynomial_features():
    """
    X = array([[0, 1],
               [2, 3],
               [4, 5]])
    """
    X = np.arange(6).reshape(3, 2)

    poly = PolynomialFeatures(2)

    poly_features = poly.fit_transform(X)

    assert_array_almost_equal(poly_features, [[0.,  1.,  0.,  0.,  1.],
                                              [2.,  3.,  4.,  6.,  9.],
                                              [4.,  5., 16., 20., 25.]])

def test_polynomial_features_degree_1():
    """Check when degree=1 array returned is same as input."""
    X = np.arange(6).reshape(3, 2)

    poly = PolynomialFeatures(1)

    poly.fit(X)

    poly_features = poly.transform(X)

    assert_array_almost_equal(poly_features, X)

def test_polynomial_features_one_feature():
    """With one feature function should features [X, X^2, X^3]."""
    X = 6 * np.random.rand(100, 1) - 3

    X_result = np.c_[X, X ** 2, X ** 3]

    poly = PolynomialFeatures(3)

    poly_features = poly.fit_transform(X)

    assert_array_almost_equal(poly_features, X_result)
