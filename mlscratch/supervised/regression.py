"""Module containing classes for supervised linear regression models."""
import numpy as np
from scipy.linalg import lstsq

from mlscratch.utils.metrics import mean_squared_error

class Regression(object):
    """
    Class representing our base regression model.  
    
    Models relationship between a dependant scaler variable y and independent
    variables X by optimizing a cost function with batch gradient descent.

    Parameters
    ----------
    n_iter : float, default=5000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-1
        Learning rate determining the size of steps in batch gradient descent.

    Attributes 
    ----------
    coef_ : array of shape (n_features, )
        Estimated coefficients for the regression problem.
    intercept_ : float 
        Bias term in the regression model.
    """
    def __init__(self, n_iter=5000, lr=1e-1):
        self.n_iter = n_iter 
        self.lr = lr 

    def fit(self, X, y):
        """
        Fit linear model with batch gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Independent variables.
        y : array-like of shape (n_samples,)
            Target values. Dependent variable.

        Returns
        -------
        self : returns an instance of self.
        """
        # Make sure inputs are numpy arrays.
        X = np.array(X)
        y = np.array(y)
        # Add x_0 = 1 to each instance for the bias term.
        X = np.c_[np.ones((X.shape[0], 1)), X]
        # Store number of samples and features in variables.
        n_samples, n_features = np.shape(X)
        self.training_errors = []
        # Initialize weights randomly from normal distribution.
        self.coef_ = np.random.randn(n_features, 1)
        # Batch gradient descent for number iterations = n_iter.
        for _ in range(self.n_iter):
            y_preds = X.dot(self.coef_)
            # Calculate mean squared error through utility function. 
            mse = mean_squared_error(y, y_preds)
            self.training_errors.append(mse)
            # Gradients of loss function.
            gradients = (2/n_samples) * X.T.dot(y_preds - y)
            # Update the weights.
            self.coef_ -= self.lr * gradients 

        return self 

    def predict(self, X):
        """
        Estimate target values using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Instances.

        Returns
        -------
        C : array of shape (n_samples,)
            Estimated targets per instance.
        """
        # Make sure inputs are numpy arrays.
        X = np.array(X)
        # Add x_0 = 1 to each instance for the bias term.
        X = np.c_[np.ones((X.shape[0], 1)), X]

        return X.dot(self.coef_)


class LinearRegression(Regression):
    """
    Class representing a linear regression model.

    Models relationship between target variable and attributes by drawing line
    that minimizes mean squared error.

    Parameters
    ----------
    n_iter : float, default=5000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-1
        Learning rate determining the size of steps in batch gradient descent.     
    solver : {'bgd', 'lstsq'}, default="bgd"
        Optimization method used to minimize mean squared error in training.

        'bgd' : 
            Batch gradient descent.

        'lstsq' : 
            Ordinary lease squares method using scipy.linalg.lstsq.

    Attributes 
    ----------
    coef_ : array of shape (n_features, )
        Estimated coefficients for the regression problem.
    intercept_ : float 
        Bias term in the regression model.

    Notes
    -----
    This class is capable of being trained using Ordinary Least Squares method
    or batch gradient descent.  See solver parameter above.
    """
    def __init__(self, n_iter=1000, lr=1e-1, solver='bgd'):
        self.solver = solver 
        super(LinearRegression, self).__init__(n_iter=n_iter, lr=lr)

    def fit(self, X, y):
        """
        Fit linear regression model.

        If solver='bgd', model is trained using batch gradient descent. 
        If solver='lstsq' model is trained using ordinary least squares.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Independent variables.
        y : array-like of shape (n_samples,)
            Target values. Dependent variable.

        Returns
        -------
        self : returns an instance of self.
        """
        # If solver is 'lsqr' use ordinary least squares optimization method.
        if self.solver == 'lstsq':
            # Make sure inputs are numpy arrays.
            X = np.array(X)
            y = np.array(y)
            # Add x_0 = 1 to each instance for the bias term.
            X = np.c_[np.ones((X.shape[0], 1)), X]
            # Scipy implementation of least squares.
            self.coef_, residues, rank, singular = lstsq(X, y)
        elif self.solver == 'bgd': 
            super(LinearRegression, self).fit(X, y)
    

    







        







        

