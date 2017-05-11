import numpy as np

import matplotlib.pyplot as plt


class AnomalyDetector(object):

    def __init__(self, multivariate=True):
        self.multivariate = multivariate
        self.mu = None
        self.sigma = None
        self.epsilon = 0.

    def train(self, X: np.matrix):
        """
        Trains the anomaly detector with a given data set. For now it is assumed that no entry in the data set correspond
        to an anomaly.
        
        :param X:   The data set
        :return:    None
        """
        m, n = X.shape

        if self.multivariate:
            self.mu = (1./m) * np.sum(X, axis=0)
            self.sigma = (1./m) * (X-self.mu).T.dot((X-self.mu))
        else:
            self.mu = np.matrix((1./m) * np.sum(X, axis=0))
            self.sigma = np.matrix(np.sqrt((1./m) * np.sum(np.square(X - self.mu), axis=0)))

        # TODO: calculate epsilon based on a set which contains anomalies
        self.epsilon = np.min(self.hypothesis(X))

    def hypothesis(self, X: np.matrix):
        """
        Calculates the hypothesis for each element in the data set.
        
        :param X:   The data set
        :return:    A Mx1 matrix with a hypothesis value for each element in the data set
        """
        m, n = X.shape

        if self.multivariate:
            # apply gaussian distribution to the data set
            p = (1./((2*np.pi)**(n/2.) * np.linalg.det(self.sigma)**(1./2))) * np.exp(-(1./2) * (X-self.mu).dot(np.linalg.inv(self.sigma).dot((X-self.mu).T)))
            p = np.matrix(p.diagonal())

            return p
        else:
            p = np.ones_like(X[:, 0])
            for j in range(0, n):
                expo = -(np.square(X[:, j] - self.mu[0, j])/(2*np.square(self.sigma[0, j])))
                # apply gaussian distribution to the data set with a different mu, sigma for each feature
                p = np.multiply(p, (1./((2*np.pi)**(1/2.) * self.sigma[0, j])) * np.exp(expo))

            return p

    def find_anomalies(self, X: np.matrix):
        """
        Finds all anomalies in a given data set by searching for elements whose hypothesis value is
        below a certain value epsilon.
        
        :param X:   The data set in which anomalies are tried to be captured
        :return:    The anomalies
        """
        p = self.hypothesis(X)

        idx = np.where((np.array(p).ravel() < self.epsilon)*1 == 1)[0]

        return X[idx, :]

    def visualize_anomalies(self, X: np.matrix, scatter=False):
        """
        Plots/Scatters the data (assumes that the x-axis values are in the first column and the y-axis values are in the
        last column) and visualizes where anomalies were detected.
        
        :param X:       The input matrix
        :param scatter: Whether the data should be scattered or plotted
        :return:        None
        """
        plt.figure(figsize=(13, 5))

        _, last_idx = X.shape
        last_idx -= 1

        if scatter:
            plt.scatter(X[:, 0], X[:, last_idx])
        else:
            plt.plot(X[:, 0], X[:, last_idx])

        anomalies = np.array(self.find_anomalies(X))

        plt.scatter(anomalies[:, 0], anomalies[:, last_idx], c='red')
        plt.show()
