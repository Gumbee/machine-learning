import numpy as np


def sum_of_squares(thetas: np.matrix, X: np.matrix, y: np.matrix, reg_lambda=1.0):
    """
    Evaluates the sum of squares cost for a given vector of parameters theta and the training set X with output y.
    
    Args:
        thetas (np.matrix):     The parameter vector used to calculate the hypothesis (Shape: 1xN)
        X (np.matrix):          The training set (Shape: MxN)
        y (np.matrix):          The training set's output (Shape: Mx1)
        reg_lambda (float):     Regularization parameter
    
    Returns:                
        float:                  sum of squares cost for the parameter vector theta on the training set X with output y.
    """
    m, n = X.shape

    hypothesis = X.dot(thetas.T)
    # theta_penalty is used to calculate the regularization penalty (excluding the first parameter)
    theta_penalty = np.zeros_like(thetas)
    theta_penalty[:, 1:] = thetas[:, 1:]

    J = (1./(2*m)) * np.sum(np.square(hypothesis-y)) + reg_lambda * np.sum(np.square(theta_penalty))

    return J


def sum_of_squares_gradient(thetas: np.matrix, X: np.matrix, y: np.matrix, reg_lambda=1.0):
    """
    Returns the sum of squares gradients of the parameter vector theta.
    
    Args:
        thetas (np.matrix):     The parameter vector used to calculate the hypothesis (Shape: 1xN)
        X (np.matrix):          The training set (Shape: MxN)
        y (np.matrix):          The training set's output (Shape: Mx1)
        reg_lambda (float):     Regularization parameter
        
    Returns:
        np.array:               The gradients of the parameters
    """
    m, n = X.shape

    hypothesis = X.dot(thetas.T)
    # theta_penalty is used to calculate the regularization penalty (excluding the first parameter)
    theta_penalty = np.zeros_like(thetas)
    theta_penalty[:, 1:] = thetas[:, 1:]

    gradients = (1./m)*((hypothesis-y).T.dot(X) + reg_lambda * theta_penalty)

    return gradients
