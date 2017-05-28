import numpy as np

from random import randint
from Training.gradient_descent import GradientDescentOptimizer as GradientDescentOptimizer
from Training.gradient_descent import GradientDescentParameters as GradientDescentParameters
from Training.accumulator import Accumulator as Accumulator


class AdaDeltaOptimizer(GradientDescentOptimizer):

    def __init__(self, batch=True, batch_size=60, epochs=5, rho=0.95, epsilon=1e-06):
        GradientDescentOptimizer.__init__(self, batch, batch_size, epochs)
        self.grd_accu = Accumulator(rho, True, epsilon)
        self.delta_accu = Accumulator(rho, True, epsilon)

    def delta(self, alpha: float, gradients):
        dlt = self.delta_accu.get_rm()
        grd = self.grd_accu.get_rm()

        if isinstance(gradients, list):
            idx = np.arange(len(gradients))
            deltas = [np.multiply(np.divide(dlt[x], grd[x]), gradients[x]) for x in idx]

            return deltas
        else:
            return np.multiply(np.divide(dlt, grd), gradients)

    def pre_update(self, gradients):
        self.grd_accu.add(gradients)

    def post_update(self, delta):
        self.delta_accu.add(delta)

    def prepare_variables(self, init_theta):
        self.grd_accu.fill_zero(init_theta)
        self.delta_accu.fill_zero(init_theta)
