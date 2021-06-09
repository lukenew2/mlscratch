"""Module containing tests for mlscratch/supervised/regression.py"""
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from mlscratch.utils.preprocessing import StandardScaler
from mlscratch.utils.preprocessing import PolynomialFeatures
from mlscratch.supervised.regression import ElasticNet, LinearRegression
from mlscratch.supervised.regression import Ridge
from mlscratch.supervised.regression import Lasso

def test_linear_regression_simple():
    """Test LinearRegression with a simple dataset."""
    X = [[1], [2]]
    y = [1, 2]

    for solver in ['bgd', 'lstsq']:
        reg = LinearRegression(solver=solver)
        reg.fit(X, y)

        assert_array_almost_equal(reg.coef_, [0., 1.])
        assert_array_almost_equal(reg.predict(X), [1., 2.])
        assert_almost_equal(reg.score(X, y), 1.0)

def test_linear_regression_complex():
    """Test LinearRegression on a more complex dataset."""
    X = np.random.rand(100, 1)
    y = 5 * X + 3
    y = np.reshape(y, (X.shape[0],))

    for solver in ['bgd', 'lstsq']:
        reg = LinearRegression(n_iter=3000, solver=solver)
        reg.fit(X, y)

        assert_array_almost_equal(reg.coef_, [3., 5.])
        assert_array_almost_equal(reg.predict(X), y)

def test_polynomial_regression():
    """
    Test LinearRegression, Ridge, Lasso, and ElasticNet using 
    PolynomialFeatures and StandardScaler classes for preprocessing.
    """
    # Preprocess data.
    X = 6 * np.random.rand(100, 1) - 3
    y = 0.5 * X ** 2 + X + 2
    y = np.reshape(y, (X.shape[0],))

    poly = PolynomialFeatures(2)

    poly_features = poly.fit_transform(X)

    scaler = StandardScaler()
    scaler.fit(poly_features)

    poly_features_scaled = scaler.transform(poly_features)

    # Test LinearRegression.
    for solver in ['bgd', 'lstsq']:
        reg = LinearRegression(n_iter=3000, solver=solver)
        reg.fit(poly_features_scaled, y)

        assert reg.coef_.shape == (poly_features_scaled.shape[1] + 1,)
        assert_array_almost_equal(reg.predict(poly_features_scaled), y)

    # Test Ridge.
    ridge = Ridge(alpha=0.)
    ridge.fit(poly_features_scaled, y)

    assert ridge.coef_.shape == (poly_features_scaled.shape[1] + 1,)
    assert_array_almost_equal(ridge.predict(poly_features_scaled), y,
                              decimal=3)

    # Test Lasso.
    lasso = Lasso(alpha=0.)
    lasso.fit(poly_features_scaled, y)

    assert lasso.coef_.shape == (poly_features_scaled.shape[1] + 1,)
    assert_array_almost_equal(lasso.predict(poly_features_scaled), y,
                              decimal=3)

    # Test ElasticNet.
    reg = ElasticNet(alpha=0.)
    reg.fit(poly_features_scaled, y)

    assert reg.coef_.shape == (poly_features_scaled.shape[1] + 1,)
    assert_array_almost_equal(reg.predict(poly_features_scaled), y, 
                              decimal=3)

def test_ridge_regression():
    """Test Ridge using score."""
    # More samples than features.
    rng = np.random.RandomState(0)

    n_samples, n_features = 6, 5
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples,)

    ridge = Ridge(alpha=1.0)

    ridge.fit(X, y)

    assert ridge.coef_.shape == (X.shape[1] + 1,)
    assert ridge.score(X, y) > 0.47
    
    # More features than samples.
    n_samples, n_features = 5, 10
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples,)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    assert ridge.score(X, y) > 0.65

def test_lasso_regression():
    """Test Lasso on simple example using various values for alpha."""

    X = [[-1], [0], [1]]
    y = [-1, 0, 1]
    T = [[2], [3], [4]]

    lasso = Lasso(n_iter=3000, alpha=1e-8)
    lasso.fit(X, y)
    pred = lasso.predict(T)

    assert_array_almost_equal(lasso.coef_, [0., 1])
    assert_array_almost_equal(pred, [2, 3, 4])

def test_lasso_regression_score():
    """Test Lasso using score."""
    # More samples than features.
    rng = np.random.RandomState(0)

    n_samples, n_features = 6, 5
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples,)

    lasso = Lasso(alpha=0.1)

    lasso.fit(X, y)

    assert lasso.coef_.shape == (X.shape[1] + 1,)
    assert lasso.score(X, y) > 0.47

    # More features than samples.
    n_samples, n_features = 5, 10
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples,)

    lasso = Lasso(alpha=0.1)

    lasso.fit(X, y)

    assert lasso.score(X, y) > 0.65


    


