import numpy as np

from Training.gradient_descent import GradientDescentOptimizer as GradientDescentOptimizer
from Training.accumulator import Accumulator as Accumulator


class AdaDeltaOptimizer(GradientDescentOptimizer):
    """
    Variation of the gradient descent optimizer implementing the ADADELTA optimizer. For a better understanding of
    the ADADELTA optimizer, take a look at the paper: https://arxiv.org/abs/1212.5701
    
    """
    def __init__(self, rho=0.95, epsilon=1e-06):
        GradientDescentOptimizer.__init__(self)
        # initialize the accumulators (not zero-filled yet!)
        self.grd_accu = Accumulator(rho, True, epsilon)
        self.delta_accu = Accumulator(rho, True, epsilon)

    def delta(self, alpha: float, gradients):
        """
        Computes the delta values by which the parameters will be changed in order to optimize the parameters values.
        
        Args:
            alpha (float):                          The learning rate
            gradients (list or np.array_like):      The gradients
        
        Returns:            
            list or np.array:                       The delta values by which the parameters should be subtracted
        """
        # get the root mean of the accumulated delta values
        dlt = self.delta_accu.get_rm()
        # get the root mean of the accumulated gradient values
        grd = self.grd_accu.get_rm()

        if isinstance(gradients, list):
            idx = np.arange(len(gradients))
            # compute the new delta value
            deltas = [np.multiply(np.divide(dlt[x], grd[x]), gradients[x]) for x in idx]

            return deltas
        else:
            return np.multiply(np.divide(dlt, grd), gradients)

    def pre_update(self, gradients):
        """
        Accumulates the gradients.
        
        Args:
            gradients (list or np.array_like): The gradient values
        """
        # accumulate the gradients
        self.grd_accu.add(gradients)

    def post_update(self, delta):
        """
        Accumulates the delta changes applied in the latest iteration.
        
        Args:
            delta (list or np.array_like): The value by which the parameters were changed.
        """
        # accumulate the delta change
        self.delta_accu.add(delta)

    def prepare_variables(self, init_theta):
        """
        Prepares the accumulator variables.
        
        Args:
            init_theta (list or np.array_like): The weights whose shape will be matched by the accumulators
        """
        # fill accumulators with zero values
        self.grd_accu.fill_zero(init_theta)
        self.delta_accu.fill_zero(init_theta)
