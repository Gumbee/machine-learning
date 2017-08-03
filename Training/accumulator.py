import numpy as np


class Accumulator(object):

    def __init__(self, decay: float = 0.95, squared=True, epsilon: float = 1e-6):
        """
        Initializes the accumulator.
        
        :param decay:       The decaying rate
        :param squared:     Whether or not to square the values when adding them to the accumulator
        :param epsilon:     A small epsilon to avoid zero values
        """
        self.decay = decay
        self.squared = squared
        self.epsilon = epsilon
        self.value = None

    def add(self, value):
        """
        Add a new value (or list of values) to the accumulator. (Accumulate a new value)
        
        :param value: The value that is accumulated
        """
        if self.squared:
            if isinstance(value, list):
                value = [np.square(x) for x in value]
            else:
                value = np.square(value)

        if isinstance(value, list):
            for i in range(0, len(value)):
                self.value[i] = np.multiply(self.decay, self.value[i]) + np.multiply((1. - self.decay), value[i])
        else:
            self.value = np.multiply(self.decay, self.value) + np.multiply((1. - self.decay), value)

    def get_rm(self):
        """
        Get the root mean value.
        
        :return: The root value of the accumulator
        """
        if isinstance(self.value, list):
            return [np.sqrt(x+self.epsilon) for x in self.value]
        else:
            return np.sqrt(self.value + self.epsilon)

    def get_val(self):
        """
        Get the value of the accumulator (for the root value, use get_rm).
        
        :return: The value of the accumulator
        """
        return self.value

    def fill_zero(self, weights):
        """
        Fills the accumulator with zero values, based on the shape of the given weights.
        
        :param weights: The weights
        :return: 
        """
        if self.value is None:
            # if we have a list of weights, allocate a list of zeros for the accumulator
            if isinstance(weights, list):
                self.value = list()
                # for each weight in the list, allocate zeros with the same shape as the weight
                for i in range(0, len(weights)):
                    self.value.append(np.zeros_like(weights[i]))
            else:
                self.value = np.zeros_like(weights)

