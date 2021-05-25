"""Module containing fixtures used across all test files."""

import pytest 

@pytest.fixture
def simple_data():
    """Fixture defining simple dataset."""
    X = [[1], [2]]
    y = [1, 2]

    return X, y 