import numpy as np


class Accumulator(object):

    def __init__(self, decay: float = 0.95, squared=True, epsilon: float = 1e-6):
        self.decay = decay
        self.squared = squared
        self.epsilon = epsilon
        self.value = None

    def add(self, value):
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
        if isinstance(self.value, list):
            return [np.sqrt(x+self.epsilon) for x in self.value]
        else:
            return np.sqrt(self.value + self.epsilon)

    def get_val(self):
        if isinstance(self.value, list):
            return self.value
        else:
            return self.value

    def fill_zero(self, model):
        if self.value is None:
            if isinstance(model, list):
                self.value = list()
                for i in range(0, len(model)):
                    self.value.append(np.zeros_like(model[i]))
            else:
                self.value = np.zeros_like(model)

