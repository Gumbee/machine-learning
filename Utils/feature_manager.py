import numpy as np
import numpy.linalg as linalg


def mean_normalization(X: np.matrix):
    """
    Calculates the mean normalization of the input matrix X.
    
    Args:
        X (np.matrix):  Input matrix
    
    Returns:
        np.matrix:      Matrix X with mean normalization
    """
    return X - X.mean(axis=0)


def standardize(X: np.matrix):
    """
    Standardizes the input matrix X.
    
    Args:
        X (np.matrix):  Input matrix
    
    Returns:
        np.matrix:      Standardized matrix X
    """
    return mean_normalization(X)/X.std(axis=0, ddof=1)


def scale_features(X: np.matrix, min_val: float = 0.0, max_val: float = 1.0):
    """ 
    Performs feature scaling on the input matrix X. Assumes that each column has a different max and min element.
    
    Args:
        X (np.matrix):      Input matrix
        min_val (float):    lower bound for the feature scaling
        max_val (float):    upper bound for the feature scaling
    
    Returns: 
        np.matrix:          Input matrix X with scaled features
    """
    Xmax = np.max(X, axis=0)
    Xmin = np.min(X, axis=0)
    return min_val + ((X - Xmin) * (max_val - min_val)) / (Xmax - Xmin)


def add_polynomial_features(X: np.matrix, degree: int = 2):
    """
    Adds polynomial features to the data.

    Args:
        X (np.matrix):  The matrix to which we add polynomial features
        degree (int):   The max degree that should be generated
    
    Returns:
        np.array:       The data with polynomial features
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

    Args:
        X (np.matrix):  The matrix to which we add inverse features
    
    Returns:
        np.array:      The data with polynomial features
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

    Args:
        X (np.matrix):  The matrix which should be projected onto k dimensions
        k (int):        The dimension onto which the data should be projected
    
    Returns:   
        np.matrix:      The input matrix X projected onto k dimensions,
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
    
    Args:
        X (np.matrix):  The input matrix X
        U (np.matrix):  The matrix U with which X is multiplied
        k (int):        The number of rows of U to use when multiplying
    
    Returns:
        np.matrix:      X multiplied with the first k rows of U
    """
    return np.matmul(X, U[:, 0:k])
