import numpy as np
import numpy.linalg as linalg


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
    Performs feature scaling on the input matrix X. Assumes that each column has a different max and min element.
    
    :param X: Input matrix
    :param min: lower bound for the feature scaling
    :param max: upper bound for the feature scaling
    :return: Input matrix X with scaled features
    """
    Xmax = np.max(X, axis=0)
    Xmin = np.min(X, axis=0)
    return min + ((X-Xmin)*(max-min))/(Xmax - Xmin)


def add_polynomial_features(X: np.matrix, degree: int = 2):
    """
    Adds polynomial features to the data.

    :param X:       The matrix to which we add polynomial features
    :param degree:  The max degree that should be generated
    :return:        The data with polynomial features
    """
    m, n = X.shape
    X_out = np.zeros((m, n*degree))
    X_out[:, 0:n] = X[:, 0:n]

    for i in range(0, m):
        for j in range(0, n):
            for k in range(2, degree + 1):
                X_out[i, (k-1)*n+j] = X[i, j]**k

    return X_out


def add_inverse(X: np.matrix):
    """
    Adds rows with the inverese of the features (x -> 1/x) to the data.

    :param X:       The matrix to which we add inverse features
    :return:        The data with polynomial features
    """
    m, n = X.shape
    X_out = np.zeros((m, n*2))
    X_out[:, 0:n] = X[:, 0:n]

    for i in range(0, m):
        for j in range(0, n):
            if X[i, j] != 0:
                X_out[i, n+j] = 1./X[i, j]
            else:
                X_out[i, n+j] = 0

    return X_out


def pca(X: np.matrix, k: int = 3):
    """
    Performs PCA and returns the projected data.

    :param X:   The matrix which should be projected onto k dimensions
    :param k:   The dimension onto which the data should be projected
    :return:    The input matrix X projected onto k dimensions,
                The eigenvectors U
    """
    m, n = X.shape

    cov_mat = (1./m) * (np.matmul(X.T, X))

    # compute singular value decomposition
    U, S, V = linalg.svd(cov_mat, full_matrices=True)

    Z = project_data(X, U, k)

    return Z, U


def project_data(X: np.matrix, U: np.matrix, k: int):
    """
    Takes an input matrix X and multiplies it with the matrix U, using only the first k rows of U.
    
    :param X: The input matrix X
    :param U: The matrix U with which X is multiplied
    :param k: The number of rows of U to use when multiplying
    :return: X multiplied with the first k rows of U
    """
    return np.matmul(X, U[:, 0:k])
