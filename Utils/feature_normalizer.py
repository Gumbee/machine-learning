import numpy as np


def mean_normalization(X: np.matrix):
    return X - X.mean()


def standardize(X: np.matrix):
    return mean_normalization(X)/X.std()


def scale_features(X: np.matrix, min: float = 0., max: float = 1.):
    Xmax = np.max(X)
    Xmin = np.min(X)
    return min + ((X-Xmin)*(max-min))/(Xmax - Xmin)
