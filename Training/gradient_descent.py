import numpy as np


class GradientDescentOptimizer(object):

    def __init__(self, learning_rate=0.1, reg_lambda=1.):
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

    def train(self, init_theta, X: np.matrix, y: np.matrix, cost_func: callable, gradient_func: callable, max_iter, debug_mode=True, **func_args):
        """
        Trains the parameters in init_theta to minimize the provided cost function.
        
        :param init_theta: The initial parameter values (if it's a list, gradient descent is applied element-wise)
        :param X: The training set
        :param y: The training set's corresponding output
        :param cost_func: The cost function that determines how good the parameters are performing
        :param gradient_func: The function that returns the parameters gradient values
        :param max_iter: Maximal number of iterations before the function should end the training
        :param debug_mode: (optional) True if debug mode should be turned on (outputs a table with important values)
        :param func_args: (optional) Additional parameters that will be passed on to cost_func and gradient_func
        :return: None
        """
        print('\nTraining Neural Network...')

        alpha = self.learning_rate
        reg_lambda = self.reg_lambda

        initial_error = cost_func(init_theta, X, y, reg_lambda, **func_args)

        if debug_mode:
            self.print_table_header('P', 'IT', 'COST', 'CHNG', 'ASCL')
            self.print_table_entry(0, 0, initial_error, initial_error, 1.00)

        # keeps track of how many entries we've already printed
        entry_num = 1
        # factor by which the learning rate alpha is scaled
        alpha_scale = 1.
        # keep track of the previous iteration's error so we can calculate the relative change
        prev_cst = initial_error

        for t in range(0, max_iter):
            # calculate gradients
            gradients = gradient_func(init_theta, X, y, reg_lambda, **func_args)

            # update weights with gradients
            # if x0 is a list, then we apply gradient descent for each item in the list
            if isinstance(init_theta, list):
                for i in range(0, len(init_theta)):
                    init_theta[i] -= alpha * np.log10(t + 1) * gradients[i]
            else:
                init_theta -= alpha * np.log10(t + 1) * gradients

            # reevaluate cost function
            cost = cost_func(init_theta, X, y, reg_lambda, **func_args)
            # get relative change of the cost function
            rel_chng = cost - prev_cst
            # update previous cost to current cost
            prev_cst = cost

            if debug_mode and t % 7 == 0:
                self.print_table_entry(entry_num, t + 1, cost, rel_chng, alpha_scale)
                entry_num += 1

        print('\033[91m', '\n{:<15s}'.format('Initial Error:'), '{:5.6e}'.format(initial_error),
              '\n{:<15s}'.format('New Error:'), '{:>5.6e}'.format(cost_func(init_theta, X, y, reg_lambda, **func_args)), '\033[0m')

    @staticmethod
    def print_table_header(First: str, Second: str, Third: str, Fourth: str, Fifth: str):
        print('\n\033[91m', '{:>4s}'.format(str(First)), '{:>1s}'.format('|'), '{:>5s}'.format(str(Second)), '{:>1s}'.format('|'),
              '{:>15s}'.format(str(Third)), '{:>1s}'.format('|'), '{:>15s}'.format(str(Fourth)), '{:>1s}'.format('|'),
              '{:>10s}'.format(str(Fifth)), '{:>1s}'.format('|'), '\033[0m')
        print('\033[91m', '{:─>63s}'.format(''), '\033[0m')

    @staticmethod
    def print_table_entry(First: int, Second: int, Third: float, Fourth: float, Fifth: float):
        print('\033[91m', '{:>4d}'.format(First), '{:1s}'.format('|'), '{:>5d}'.format(Second), '{:>1s}'.format('|'),
              '{:>15.6e}'.format(Third), '{:>1s}'.format('|'), '{:>15.6e}'.format(Fourth), '{:>1s}'.format('|'),
              '{:>10.3f}'.format(Fifth), '{:>1s}'.format('|'), '\033[0m')
