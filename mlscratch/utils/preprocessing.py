"""Module containing classes for preprocessing data."""
from itertools import combinations_with_replacement
import numpy as np 
from scipy.special import factorial

class StandardScaler:
    """
    Standardize features by centering the mean to 0 and unit variance.

    The standard score of an instance is calculated by:

        z = (x - u) / s

    where u is the mean of the training data and s is the standard deviation.

    Standardizing data is often necessary before training many machine
    learning models to avoid problems like exploding/vanishing gradients and
    feature dominance.

    Attributes
    ----------
    mean_ : numpy array of shape (n_features, )
        The mean of each feature in the training set.
    var_ : numpy array of shape (n_features, )
        The variance of each feature in the training set.
    """
    def fit(self, X):
        """
        Calculate and store the mean and variance of each feature in the
        training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data set to calculate mean and variance of.
        """
        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)


    def transform(self, X):
        """
        Standardize data by subtracting out the mean and dividing by
        standard deviation calculated during fitting.

        Parameters 
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be standardized

        Returns 
        -------
        X_std : array-like of shape(n_samples, n_featuers)
            Standardized data.
        """
        X_std = (X - self.mean_) / np.sqrt(self.var_)

        return X_std

    def inverse_transform(self, X_std):
        """
        Transform data back into orginal state by multiplying by standard
        deviation and adding the mean back in.

        Inverse standard scaler:

            x = z * s + u

        where s is the standard deviation, and u is the mean.

        Parameters 
        ----------
        X_std : array-like of shape (n_samples, n_features)
            Standardized data to convert into original state.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        X = X_std * np.sqrt(self.var_) + self.mean_


        return X 


class PolynomialFeatures:
    """
    Generate polynomial features. 

    Generate a new matrix of features including all combinations of different
    polynomial features less than or equal to the specified degree. No bias 
    term is calculated in this implementation because our regression models
    create a bias term when they are trained.

    Parameters
    ----------
    degree : int, default=2
        Degree of polynomial features to be created.

    Attributes
    ----------
    n_input_features : int 
        The total number of input features.
    n_output_features : int
        The total number of output features computed by iterating over all
        possible polynomial combinations of input features.
    """
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X):
        """
        Compute the number of output features.

            n_output_features = (n+d)!/d!n!

        where n is the number of input features and d is the degree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix to be transformed into polynomial feature matrix.
        """
        # Make sure input is numpy array.
        X = np.array(X)
        # Store number of input features in attribute.
        self.n_input_features = X.shape[1]
        # Calculate numerator and denominator of equation listed above.
        numerator = factorial(self.n_input_features + self.degree)
        denominator = factorial(self.degree) * factorial(self.n_input_features)
        # Calculate number of output features minus 1 to subtract bias term.
        self.n_output_features = int(numerator / denominator) - 1


    def transform(self, X):
        """
        Transform data to polynomial feature matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix to be transformed into polynomial feature matrix.

        Returns
        -------
        X : array-like of shape (n_samples, n_output_features)
            Tranformed polynomial feature matrix where n_output_features is
            the number of output features after polynomial transformation.
        """
        # Generate all combination of feature indices.
        combos = [combinations_with_replacement(range(self.n_input_features),i)
                  for i in range(1, self.degree + 1)]
        # Create list of tuples containing feature index combinations.
        combinations = [item for sublist in combos for item in sublist]
        # Create new array of the desired output shape.
        X_new = np.empty((X.shape[0], self.n_output_features))
        # Multiply features for each combination tuple in combinations.
        for i, index_combos in enumerate(combinations):
            X_new[:, i] = np.prod(X[:, index_combos], axis=1)

        return X_new

    def fit_transform(self, X):
        """
        Compute the number of output features and transform data to polynomial
        feature matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix to be transformed into polynomial feature matrix.

        Returns 
        -------
        X : array-like of shape (n_samples, n_output_features)
            Transformed polynomial feature matrix where n_output_features is 
            the number of output features after polynomial transformation.
        """
        self.fit(X)

        return self.transform(X)












