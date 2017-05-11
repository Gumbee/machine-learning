import numpy as np


def mean_normalization(X: np.matrix):
    """
    Calculates the mean normalization of the input matrix X.
    
    :param X: Input matrix
    :return: Matrix X with mean normalization
    """
    return X - X.mean(axis=0)


def standardize(X: np.matrix):
    """
    Standardizes the input matrix X.
    
    :param X: Input matrix
    :return: Standardized matrix X
    """
    return mean_normalization(X)/X.std(axis=0, ddof=1)


def scale_features(X: np.matrix, min: float = 0., max: float = 1.):
    """ 
    Performs feature scaling on the input matrix X.
    
    :param X: Input matrix
    :param min: lower bound for the feature scaling
    :param max: upper bound for the feature scaling
    :return: Input matrix X with scaled features
    """
    Xmax = np.max(X, axis=0)
    Xmin = np.min(X, axis=0)
    return min + ((X-Xmin)*(max-min))/(Xmax - Xmin)
