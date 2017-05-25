import numpy as np

from Training.gradient_descent import GradientDescentOptimizer as GradientDescentOptimizer
from Training.accumulator import Accumulator as Accumulator


class AdaDeltaOptimizer(GradientDescentOptimizer):

    def __init__(self, batch=True, batch_size=60, epochs=5, rho=0.95, epsilon=1e-06):
        GradientDescentOptimizer.__init__(self, batch, batch_size, epochs)
        self.grd_accu = Accumulator(rho, True, epsilon)
        self.delta_accu = Accumulator(rho, True, epsilon, printer=True)

    def delta(self, alpha: float, gradients):
        dlt = self.delta_accu.get_rm()
        grd = self.grd_accu.get_rm()

        if isinstance(gradients, list):
            deltas = list()

            for i in range(0, len(gradients)):
                deltas.append(np.multiply(-1, np.multiply(np.divide(dlt[i], grd[i]), gradients[i])))

            return deltas
        else:
            return np.multiply(-1, np.multiply(np.divide(dlt, grd), gradients))

    def pre_update(self, gradients):
        i, j = (0, 2)
        # print("Gradient:", gradients[0][i, j])
        self.grd_accu.add(gradients)
        # print("AccumGrad:", self.grd_accu.get_rm()[0][i, j])

    def post_update(self, delta):
        i, j = (0, 2)
        # print("Delta:", delta[0][i, j])
        # print("AccumDelta:", self.updt_accu.get_rm()[0][i, j])
        self.delta_accu.add(delta)

    def prepare_variables(self, init_theta):
        self.grd_accu.fill_zero(init_theta)
        self.delta_accu.fill_zero(init_theta)
