import time


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
    epochs = 10
    batch_size = 32
    learning_rate = 0.1
    reg_lambda = 1.
    cost_func = None
    gradient_func = None
    func_args = {}
    callback = None
    callback_args = {}


class GDLoggingParameters(object):
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
