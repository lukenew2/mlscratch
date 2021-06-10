"""Module containing tests for mlscratch/utils/regularization.py."""
import numpy as np
from numpy.testing import assert_almost_equal

from mlscratch.utils.regularization import l1_regularization
from mlscratch.utils.regularization import l2_regularization
from mlscratch.utils.regularization import l1_l2_regularization

def test_l1_regularization():

    w = [[2], [1], [2]]

    regularization = l1_regularization(alpha=0.5)

    assert_almost_equal(regularization(w[1:]), 1.5)
    assert_almost_equal(regularization.grad(w[1:]), [[0.], [0.5], [0.5]])

def test_l2_regularization():

    w = [[2], [1], [1]]

    regularization = l2_regularization(alpha=0.5)

    assert_almost_equal(regularization(w[1:]), 0.25 * np.sqrt(2))
    assert_almost_equal(regularization.grad(w[1:]), [[0.], [0.5], [0.5]])

def test_l1_l2_regularization():

    # Test equivalence to l2 regularization when r = 0
    w = [[2], [1], [1]]

    regularization = l1_l2_regularization(alpha=0.5, r=0)

    assert_almost_equal(regularization(w[1:]), 0.25 * np.sqrt(2))
    assert_almost_equal(regularization.grad(w[1:]), [[0.], [0.5], [0.5]])

    # Test equivalence to l1 regularization when r = 1
    w = [[2], [1], [2]]

    regularization = l1_l2_regularization(alpha=0.5, r=1)

    assert_almost_equal(regularization(w[1:]), 1.5)
    assert_almost_equal(regularization.grad(w[1:]), [[0.], [0.5], [0.5]])

    # Test Elastic Net for correct output when r = 0.5
    w = [[0], [1]]

    regularization = l1_l2_regularization(alpha=0.5, r=0.5)

    assert_almost_equal(regularization(w[1:]), .375)
    assert_almost_equal(regularization.grad(w[1:]), [[0.], [0.5]])