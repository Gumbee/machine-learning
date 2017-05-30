import numpy as np

from Logging.logger import GDLoggingParameters as GDLoggingParameters
from Logging.logger import LogHandler as LogHandler
from parameters import GradientDescentParameters as GradientDescentParameters


class GradientDescentOptimizer(object):

    def train(self, init_theta, X: np.matrix, y: np.matrix, gd_parameters: GradientDescentParameters, log_handler: LogHandler = None):
        """
        Trains the parameters in init_theta to minimize the provided cost function.
        
        :param init_theta:      The initial parameter values (if it's a list, gradient descent is applied element-wise)
        :param X:               The training set
        :param y:               The training set's corresponding output
        :param gd_parameters:   The parameters with which gradient descent should be run
        :param log_handler:     The LogHandler instance which handles all the logging
        :return:                None
        """
        print('\nTraining Parameters...')

        log_handler = log_handler or LogHandler()

        # get relevant gradient descent parameters
        epochs = gd_parameters.epochs
        batch_size = gd_parameters.batch_size
        alpha = gd_parameters.learning_rate
        cost_func = gd_parameters.cost_func
        gradient_func = gd_parameters.gradient_func
        reg_lambda = gd_parameters.reg_lambda
        func_args = gd_parameters.func_args
        callback = gd_parameters.callback
        callback_args = gd_parameters.callback_args

        # remember with which error rate we started
        initial_error = gd_parameters.cost_func(init_theta, X, y, reg_lambda)

        m, _ = X.shape

        session_id = log_handler.open_gd_session(initial_error)

        self.prepare_variables(init_theta)

        idx = np.random.permutation(m)

        for i in range(0, epochs):

            # train all the batches
            for x in range(0, m, batch_size):
                # determine at which index the current batch ends
                end = min(x+batch_size, m-1)

                # calculate gradients
                gradients = gradient_func(init_theta, X[idx[x:end], :], y[idx[x:end], :], reg_lambda, **func_args)

                # perform pre-update calculations if necessary
                self.pre_update(gradients)

                # get the values by which we change our parameters
                delta = self.delta(alpha, gradients)

                # update weights with our delta values
                # if init_theta is a list, then we apply gradient descent for each item in the list (e.g Neural Networks)
                if isinstance(init_theta, list):
                    for e in range(0, len(init_theta)):
                        init_theta[e] -= delta[e]
                else:
                    init_theta -= delta

                # perform post-update calculations if necessary
                self.post_update(delta)

                log_handler.log_gd_progress(session_id, i, x, init_theta, X, y, gd_parameters)

                if callback is not None:
                    callback(init_theta, X, i, **callback_args)

        print('\033[91m', '\n{:<15s}'.format('Initial Error:'), '{:5.6e}'.format(initial_error),
              '\n{:<15s}'.format('New Error:'),
              '{:>5.6e}'.format(cost_func(init_theta, X, y, reg_lambda, **func_args)), '\033[0m')

        log_handler.close_gd_session()

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


