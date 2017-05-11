import numpy as np


class GradientDescentParameters(object):
    """
    Contains all necessary settings for the gradient descent algorithm.
    
    learning_rate:      Gradient descent's factor by which the gradient is applied
    reg_lambda:         Regularization factor
    cost_func:          The cost function that determines how good the parameters are performing
    gradient_func:      The function that returns the parameters gradient values
    max_iter:           Maximum number of iterations before the function should end the training
    debug_mode:         (optional) True if debug mode should be turned on (outputs a table with important values). Default: True
    func_args:          (optional) Additional parameters that will be passed on to cost_func and gradient_func
    callback:           (optional) Function that is called after every iteration with the training set, the current parameter values
                        and the current iteration number as parameters. Additional parameters can be specified via callback_args
    callback_args:      (optional) Additional parameters that should be passed to the callback function
    """
    learning_rate = 0.1
    reg_lambda = 1.
    cost_func: callable = None
    gradient_func: callable = None
    func_args: dict = {}
    max_iter: int = 6
    debug_mode: bool = True
    callback: callable = None
    callback_args: dict = {}


class GradientDescentOptimizer(object):

    def __init__(self, batch=True, batch_size: int = 60, epochs=5):
        self.batch = batch
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, init_theta, X: np.matrix, y: np.matrix, gd_parameters: GradientDescentParameters):
        """
        Trains the parameters in init_theta to minimize the provided cost function.
        
        :param init_theta:      The initial parameter values (if it's a list, gradient descent is applied element-wise)
        :param X:               The training set
        :param y:               The training set's corresponding output
        :param gd_parameters:   The parameters with which gradient descent should be run
        :return:                None
        """
        print('\nTraining Parameters...')

        reg_lambda = gd_parameters.reg_lambda
        cost_func = gd_parameters.cost_func
        func_args = gd_parameters.func_args
        debug_mode = gd_parameters.debug_mode
        callback = gd_parameters.callback
        callback_args = gd_parameters.callback_args

        initial_error = gd_parameters.cost_func(init_theta, X, y, reg_lambda)

        # keeps track of how many entries we've already printed
        entry_num = 1
        # factor by which the learning rate alpha is scaled
        alpha_scale = 1.
        # keep track of the previous iteration's error so we can calculate the relative change
        prev_cst = initial_error
        # keep track of how often we didn't change the cost by applying a gradient descent step
        num_converged = 0

        m, _ = X.shape

        if debug_mode:
            self.print_table_header('P', 'IT', 'COST', 'CHNG', 'ASCL')
            self.print_table_entry(0, 0, initial_error, initial_error, 1.00)

        idx = np.random.permutation(m)
        for i in range(0, self.epochs):

            for x in range(0, m, self.batch_size):
                end = min(x+self.batch_size, m-1)
                self.train_batch(init_theta, X[idx[x:end], :], y[idx[x:end], :], gd_parameters)

            # reevaluate cost function
            cost = cost_func(init_theta, X, y, reg_lambda, **func_args)
            # get relative change of the cost function
            rel_chng = cost - prev_cst
            # update previous cost to current cost
            prev_cst = cost

            if debug_mode and i % 1 == 0:
                self.print_table_entry(entry_num, i + 1, cost, rel_chng, alpha_scale)
                entry_num += 1

            if rel_chng - (-1e-30) > 0:
                if num_converged > 50:
                    print('\n\033[91mGradient Descent converged. Training ended.\033[0m')
                    return
                else:
                    num_converged += 1
            else:
                num_converged = 0

            if callback is not None:
                callback(init_theta, X, t, **callback_args)

        print('\033[91m', '\n{:<15s}'.format('Initial Error:'), '{:5.6e}'.format(initial_error),
              '\n{:<15s}'.format('New Error:'),
              '{:>5.6e}'.format(cost_func(init_theta, X, y, reg_lambda, **func_args)), '\033[0m')

    def train_batch(self, init_theta, X: np.matrix, y: np.matrix, gd_parameters: GradientDescentParameters):
        # retrieve parameter values
        alpha = gd_parameters.learning_rate
        reg_lambda = gd_parameters.reg_lambda
        cost_func = gd_parameters.cost_func
        gradient_func = gd_parameters.gradient_func
        func_args = gd_parameters.func_args
        max_iter = gd_parameters.max_iter
        debug_mode = gd_parameters.debug_mode
        callback = gd_parameters.callback
        callback_args = gd_parameters.callback_args

        self.prepare_variables(init_theta)

        for t in range(0, max_iter):
            # calculate gradients
            gradients = gradient_func(init_theta, X, y, reg_lambda, **func_args)

            self.pre_update(gradients)

            delta = self.delta(alpha, gradients)

            # update weights with gradients
            # if x0 is a list, then we apply gradient descent for each item in the list
            if isinstance(init_theta, list):
                for i in range(0, len(init_theta)):
                    init_theta[i] += delta[i]
            else:
                init_theta += delta

            self.post_update(delta)

    def delta(self, alpha: float, gradients):
        if isinstance(gradients, list):
            for i in range(0, len(gradients)):
                gradients[i] *= -alpha
            return gradients
        else:
            return -alpha * gradients

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