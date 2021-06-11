"""Module containing classes for logistic and softmax regression."""
import numpy as np

from mlscratch.utils.activations import Sigmoid
from mlscratch.utils.activations import Softmax
from mlscratch.utils.preprocessing import OneHotEncoder

class LogisticRegression():
    """
    Class representing a logistic regression model.

    Determines the porbability of an instance belonging to the positive
    class in binary classification problems. 

    Parameters
    ----------
    n_iter : float, default=3000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-1
        Learning rate determining the size of steps in batch gradient descent.

    Attributes 
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients of each feature and intercept.
    """
    def __init__(self, n_iter=3000, lr=1e-1):
        self.n_iter = n_iter
        self.lr = lr
        self.sigmoid = Sigmoid()

    def fit(self, X, y):
        """
        Fit logistic regression model.

        This implementation uses batch gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Independent variables.
        y : array-like of shape (n_samples,)
            Target classes. Dependent variable.

        Returns
        -------
        self : returns an instance of self.
        """
        # Insert X_0 = 1 for the bias term.
        X = np.insert(X, 0, 1, axis=1)

        # Store number of features in variables.
        n_features = X.shape[1]

        # Randomly intialize weights using glorot uniform intializer.
        limit = np.sqrt(2 / n_features)
        self.coef_ = np.random.uniform(-limit, limit, (n_features,))

        # Perform n_iter number of iterations of batch gradient descent.
        for _ in range(self.n_iter):

            # Calculate the logistic of predictions.
            y_preds = self.sigmoid(X.dot(self.coef_))

            # Calculate gradients of cost function.
            gradients = X.T.dot(y_preds - y)

            # Update the weights.
            self.coef_ -= self.lr * gradients 

        return self

    def predict(self, X):
        """
        Estimate target values after model has been fitted.

            0 if p < 0.5
            1 if p >= 0.5

        predicts 1 for the positive class or 0 for the negative class.

        Parameters
        ----------
        X: array-like of shape(n_samples, n_features)
            Feature matrix.

        Returns
        -------
        C: array of shape(n_samples,)
            Estimated classes.
        """
        # Insert X_0 = 1 for the bias term.
        X = np.insert(X, 0, 1, axis=1)

        # Estimate target classes.
        y_preds = np.round(self.sigmoid(X.dot(self.coef_))).astype(int)

        return y_preds

    def predict_proba(self, X):
        """
        Estimated probabilities of instances belonging to the positive class.

        Parameters
        ----------
        X: array-like of shape(n_samples, n_features)
            Feature matrix.

        Returns
        -------
        C: array of shape (n_samples,)
            Estimated probabilities.
        """
        # Insert X_0 = 1 for the bias term.
        X = np.insert(X, 0, 1, axis=1)

        # Estimate probabilities of instances belonging to positive class.
        probas = self.sigmoid(X.dot(self.coef_))

        return probas


class SoftmaxRegression():
    """
    Class representing a softmax regression model.

    Capable of performing multiclass classfication.

    Parameters
    ----------
    n_iter : float, default=3000
        Maximum number of iterations to be used by batch gradient descent.
    lr : float, default=1e-1
        Learning rate determining the size of steps in batch gradient descent.

    Attributes 
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients of each feature and intercept.  
    """
    def __init__(self, n_iter=3000, lr=1e-1):
        self.n_iter = n_iter
        self.lr = lr
        self.encoder = OneHotEncoder()
        self.softmax = Softmax()

    def fit(self, X, y):
        """
        Fit softmax regression model.

        This implementation uses batch gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Independent variables.
        y : array-like of shape (n_samples,)
            Target classes. Dependent variable.

        Returns
        -------
        self : returns an instance of self.
        """
        # Insert X_0 = 1 for the bias term.
        X = np.insert(X, 0, 1, axis=1)

        # Store number of features and classes in variables.
        n_features = X.shape[1]
        n_classes = len(np.unique(y))

        # Randomly intialize weights using glorot uniform intializer.
        limit = np.sqrt(2 / n_features)
        self.coef_ = np.random.uniform(-limit, limit, (n_features, n_classes))

        # Perform n_iter number of iterations of batch gradient descent.
        for _ in range(self.n_iter):
            
            # Estimated probabilities of instances belonging to each class.
            y_preds = self.softmax(X.dot(self.coef_))

            # Gradients of cost function.
            gradients = X.T.dot(y_preds - self.encoder(y))

            # Update the weights.
            self.coef_ -= self.lr * gradients 

        return self

    def predict(self, X):
        """
        Estimate target classes of feature matrix.

        Parameters
        ----------
        X: array-like of shape(n_samples, n_features)
            Feature matrix.

        Returns
        -------
        C: array of shape(n_samples,)
            Estimated classes.
        """
        # Insert X_0 = 1 for the bias term.
        X = np.insert(X, 0, 1, axis=1)

        # Estimated probabilities of instances belonging to each class.
        probas = self.softmax(X.dot(self.coef_))

        # Return class of highest probability
        return np.argmax(probas, axis=1)

    def predict_proba(self, X):
        """
        Estimated probabilities of instances belonging to each class.

        Parameters
        ----------
        X: array-like of shape(n_samples, n_features)
            Feature matrix.

        Returns
        -------
        C: array of shape (n_samples, n_classes)
            Estimated probabilities.
        """
        # Insert X_0 = 1 for the bias term.
        X = np.insert(X, 0, 1, axis=1)

        # Estimated probabilities of instances belonging to each class.
        probas = self.softmax(X.dot(self.coef_))

        return probas
