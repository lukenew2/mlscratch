"""Module containing classes for regularization of linear models."""
import numpy as np

class l1_regularization():
    """
    Add l1 regularization penalty to linear models.

    Regularization term:

        alpha * ||w||

    where w is the vector of feature weights and alpha is the hyperparameter
    controlling how much regularization is done to the model.

    Parameters
    ----------
    alpha : float, default=1.0
        Factor determining the amount of regularization to be performed on
        the model.

    Notes
    -----
    The bias term is not regularized and therefore should be omitted from the
    feature weights as input.  
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha 

    def __call__(self, w):
        "Calculate l1 regularization term."
        return self.alpha * np.linalg.norm(w, 1)

    def grad(self, w):
        """Calculate subgradient vector of l1 regularization penalty.
        
                      -1 if w_i < 0
            sign(w) =  0 if w_i = 0
                       1 if w_i > 0

        where w is the vector of feature weights.
        """
        subgradient = self.alpha * np.sign(w)
        
        # Insert 0 for bias term.
        return np.insert(subgradient, 0, 0, axis=0)

class l2_regularization():
    """
    Add l2 regularization penalty to linear models.

    Regularization term:

        alpha * 1/2 * ||w||^2

    Where w is the vector of feature weights and alpha is the hyperparameter
    controlling how much regularization is done to the model.

    Parameters
    ----------
    alpha : float, default=1.0
        Factor determining the amount of regularization to be performed on
        the model.

    Notes
    -----
    The bias term is not regularized and therefore should be omitted from the
    feature weights as input.  
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha 

    def __call__(self, w):
        """Calculate regularization term."""
        return self.alpha * 0.5 * np.linalg.norm(w, 2)
    
    def grad(self, w):
        """
        Calculate gradient descent regularization term.

            alpha * w

        where alpha is the factor determining the amount of regularization and
        w is the vector of feature weights.  
        """
        gradient_penalty = np.asarray(self.alpha) * w

        # Insert 0 for bias term.
        return np.insert(gradient_penalty, 0, 0, axis=0)


class l1_l2_regularization():
    """
    Add a mix of l1 and l2 regularization penalty to linear models.

    Regularization term:

        r * alpha * ||w|| + (1 - r) / 2 * alpha * ||w||^2

    where r is the mix ratio, alpha is the factor determining the amount
    of regularization and w is the vector of feature weights.

    Parameters
    ----------
    alpha : float, default=1.0
        Factor determining the amount of regularization to be performed on
        the model.
    r : float, default=0.5
        Mix ratio determining the amount of l1 vs l2 regularization to add.  
        A value of 0 is equivalent to l2 regularization and a value of 1 is
        equivalent to l1 regularization.

    Notes
    -----
    The bias term is not regularized and therefore should be omitted from the
    feature weights as input.  
    """
    def __init__(self, alpha=1.0, r=0.5):
        self.alpha = alpha
        self.r = r 

    def __call__(self, w):
        """Calculate elastic net regularization penalty."""
        l1_term = self.alpha * np.linalg.norm(w, 1)
        l2_term = self.alpha * 0.5 * np.linalg.norm(w, 2)

        return self.r * l1_term + (1 - self.r) * l2_term

    def grad(self, w):
        """
        Calculate gradient descent regularization penalty.

            alpha * (r * sign(w) + (1 - r) * w)

        where r is the mix ratio, alpha is the factor determining the amount
        of regularization and w is the vector of feature weights.
        """
        l1_grad = self.r * np.sign(w)
        l2_grad = np.asarray(1 - self.r) * w 

        gradient_penalty = self.alpha * (l1_grad + l2_grad)

        # Insert 0 for bias term.
        return np.insert(gradient_penalty, 0, 0, axis=0)