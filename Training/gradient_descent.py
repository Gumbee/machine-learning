import numpy as np
import time

from os import path as os_path
from os import makedirs as os_makedirs
from definitions import ROOT_DIR


class GradientDescentParameters(object):
    """
    Contains all necessary settings for the gradient descent algorithm.

    learning_rate:      Gradient descent's factor by which the gradient is applied
    reg_lambda:         Regularization factor
    cost_func:          The cost function that determines how good the parameters are performing
    gradient_func:      The function that returns the parameters gradient values
    debug_mode:         (optional) True if debug mode should be turned on (outputs a table with important values). Default: True
    func_args:          (optional) Additional parameters that will be passed on to cost_func and gradient_func
    callback:           (optional) Function that is called after every iteration with the training set, the current parameter values
                        and the current iteration number as parameters. Additional parameters can be specified via callback_args
    callback_args:      (optional) Additional parameters that should be passed to the callback function
    """
    learning_rate = 0.1
    reg_lambda = 1.
    cost_func = None
    gradient_func = None
    func_args = {}
    callback = None
    callback_args = {}


class LoggingParameters(object):
    """
    Contains all necessary settings for the gradient descent algorithm to know how to log the progress.
    """
    log_progress = True
    num_cost_evaluations = 50
    cost_eval_use_subset = True
    cost_eval_subset_size = 5000
    log_file_name = 'gd_log_'

    def __init__(self):
        self.log_file_name = 'gd_log_' + time.strftime("%Y%m%d-%H%M%S")



class GradientDescentOptimizer(object):

    def __init__(self, batch=True, batch_size: int = 60, epochs=5):
        self.batch = batch
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, init_theta, X: np.matrix, y: np.matrix, gd_parameters: GradientDescentParameters, log_parameters: LoggingParameters = None):
        """
        Trains the parameters in init_theta to minimize the provided cost function.
        
        :param init_theta:      The initial parameter values (if it's a list, gradient descent is applied element-wise)
        :param X:               The training set
        :param y:               The training set's corresponding output
        :param gd_parameters:   The parameters with which gradient descent should be run
        :param log_parameters:  The parameters which specify how the progress should be logged
        :return:                None
        """
        print('\nTraining Parameters...')

        if log_parameters is None:
            log_parameters = LoggingParameters()

        alpha = gd_parameters.learning_rate
        cost_func = gd_parameters.cost_func
        gradient_func = gd_parameters.gradient_func
        reg_lambda = gd_parameters.reg_lambda
        func_args = gd_parameters.func_args
        callback = gd_parameters.callback
        callback_args = gd_parameters.callback_args

        log_progress = log_parameters.log_progress
        num_cost_eval = log_parameters.num_cost_evaluations
        cost_eval_use_subset = log_parameters.cost_eval_use_subset
        cost_eval_subset_size = log_parameters.cost_eval_subset_size
        log_file_name = log_parameters.log_file_name

        initial_error = gd_parameters.cost_func(init_theta, X, y, reg_lambda)

        # keeps track of how many entries we've already printed
        entry_num = 1
        # keep track of the previous iteration's error so we can calculate the relative change
        prev_cst = initial_error
        # keep track of how often we didn't change the cost by applying a gradient descent step

        m, _ = X.shape

        if log_progress:
            self.print_table_header('P', 'EP', 'COST', 'CHNG', 'ASCL')
            self.print_table_entry(0, 0, initial_error, initial_error, 1.00)

        self.prepare_variables(init_theta)

        idx = np.random.permutation(m)

        for i in range(0, self.epochs):

            # train all the batches
            for x in range(0, m, self.batch_size):
                end = min(x+self.batch_size, m-1)
                # calculate gradients
                gradients = gradient_func(init_theta, X[idx[x:end], :], y[idx[x:end], :], reg_lambda, **func_args)

                self.pre_update(gradients)

                delta = self.delta(alpha, gradients)

                # update weights with gradients
                # if x0 is a list, then we apply gradient descent for each item in the list
                if isinstance(init_theta, list):
                    for e in range(0, len(init_theta)):
                        init_theta[e] -= delta[e]
                else:
                    init_theta -= delta

                self.post_update(delta)

                # only log the progress and reevaluate the cost every other iteration
                if log_progress and x/self.batch_size % int(max((m/self.batch_size)/num_cost_eval, 1)) == 0:
                    if cost_eval_use_subset:
                        cst_idx = np.random.permutation(m)
                        cost_eval_subset_size = min(cost_eval_subset_size, m)
                        cst_idx = cst_idx[0:cost_eval_subset_size]

                        # reevaluate cost function
                        cost = cost_func(init_theta, X[cst_idx, :], y[cst_idx, :], reg_lambda, **func_args)
                    else:
                        cost = cost_func(init_theta, X, y, reg_lambda, **func_args)

                    # get relative change of the cost function
                    rel_chng = cost - prev_cst
                    # update previous cost to current cost
                    prev_cst = cost
                    self.print_table_entry(entry_num, i + 1, cost, rel_chng, 1.0)
                    self.write_progress_to_file(entry_num, i + 1, cost, rel_chng, log_file_name)
                    entry_num += 1

                if callback is not None:
                    callback(init_theta, X, i, **callback_args)

        print('\033[91m', '\n{:<15s}'.format('Initial Error:'), '{:5.6e}'.format(initial_error),
              '\n{:<15s}'.format('New Error:'),
              '{:>5.6e}'.format(cost_func(init_theta, X, y, reg_lambda, **func_args)), '\033[0m')

    def delta(self, alpha: float, gradients):
        if isinstance(gradients, list):
            for i in range(0, len(gradients)):
                gradients[i] *= alpha
            return gradients
        else:
            return alpha * gradients

    def pre_update(self, gradients):
        pass

    def post_update(self, delta):
        pass

    def prepare_variables(self, init_theta):
        pass

    # ================= Verification Functions =================

    @staticmethod
    def check_gradients(theta: np.matrix, X: np.matrix, y: np.matrix, gradients, gd_parameters: GradientDescentParameters,  epsilon=1e-4, threshold=1e-6):
        """
        Numerically calculate the gradients based on the current model and compare them to the given gradients. 
        If they don't match, raise an error.

        :param theta:           Parameter values
        :param X:               The training set on which the model was trained
        :param y:               The output corresponding to the training set
        :param gradients:       The gradients which are to be checked
        :param gd_parameters:   The gradient descent's settings
        :param epsilon:         (optional) How accurate the numerical gradient should be (the smaller the better, but beware too small values)
        :param threshold:       (optional) If the difference between the numerical gradient and the provided gradient is
                                bigger than the threshold an error will be raised
        :return:                None
        """

        cost_func = gd_parameters.cost_func
        func_args = gd_parameters.func_args
        reg_lambda = gd_parameters.reg_lambda

        if isinstance(gradients, np.matrix):
            n = len(theta)
            for j in range(0, n):
                # store the initial weight
                initial_weight = theta[0, j]
                # add a small value to the initial weight
                theta[0, j] = initial_weight + epsilon
                # calculate the new cost function with the small value added to the weight element
                plus = cost_func(theta, X, y, reg_lambda, **func_args)
                # subtract a small value from the initial weight
                theta[0, j] = initial_weight - epsilon
                # calculate the new cost function with the small value subtracted to the weight element and save
                # the difference between the cost where we added a value and the cost where we subtracted it
                num_grad = (plus - cost_func(theta, X, y, reg_lambda, **func_args)) / (2 * epsilon)
                # restore the weight element's initial weight
                theta[0, j] = initial_weight
                if gradients[0, j] - num_grad > threshold:
                    print('Numerical:', num_grad)
                    print('Algorithm:', gradients[0, j])
                    # raise an error if the difference between the numerical gradient and the provided gradient
                    # is exceeding the threshold
                    raise Exception('Gradients do not match!')

        elif isinstance(gradients, list) and isinstance(theta, list):
            for w in range(0, len(gradients)):
                m, n = gradients[w].shape
                # loop through all gradients
                for i in range(0, m):
                    for j in range(0, n):
                        # store the initial weight
                        initial_weight = theta[w][i, j]
                        # add a small value to the initial weight
                        theta[w][i, j] = initial_weight + epsilon
                        # calculate the new cost function with the small value added to the weight element
                        plus = cost_func(theta, X, y, reg_lambda, **func_args)
                        # subtract a small value from the initial weight
                        theta[w][i, j] = initial_weight - epsilon
                        # calculate the new cost function with the small value subtracted to the weight element and save
                        # the difference between the cost where we added a value and the cost where we subtracted it
                        num_grad = (plus - cost_func(theta, X, y, reg_lambda, **func_args)) / (2 * epsilon)
                        # restore the weight element's initial weight
                        theta[w][i, j] = initial_weight
                        if gradients[w][i, j] - num_grad > threshold:
                            print('Numerical:', num_grad)
                            print('Algorithm:', gradients[w][i, j])
                            # raise an error if the difference between the numerical gradient and the provided gradient
                            # is exceeding the threshold
                            raise Exception('Gradients do not match!')
        else:
            raise Exception('Unknown type of gradients!')

    # ================= Util Functions =================

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

    @staticmethod
    def write_progress_to_file(iteration: int, epoch: int, cost: float, rel_chng: float, file_name: str):
        path = os_path.join(ROOT_DIR, 'Logs/' + file_name)

        if not os_path.exists(os_path.dirname(path)):
            os_makedirs(os_path.dirname(path))

        log_file = open(path, 'a')
        print(iteration, epoch, cost, rel_chng, file=log_file)

