import time
import numpy as np


class GradientDescentParameters(object):
    """
    Contains all necessary settings for the gradient descent algorithm.

    epochs:             How many epochs of training
    batch_size:         Size of the batches that are trained
    learning_rate:      Gradient descent's factor by which the gradient is applied
    reg_lambda:         Regularization factor
    cost_func:          The cost function that determines how good the parameters are performing
    gradient_func:      The function that returns the parameters gradient values
    func_args:          (optional) Additional parameters that will be passed on to cost_func and gradient_func
    callback:           (optional) Function that is called after every iteration with the training set, the current parameter values
                        and the current iteration number as parameters. Additional parameters can be specified via callback_args
    callback_args:      (optional) Additional parameters that should be passed to the callback function
    """
    epochs = 2
    batch_size = 32
    learning_rate = 0.1
    reg_lambda = 0.
    cost_func = None
    gradient_func = None
    func_args = {}
    callback = None
    callback_args = {}


class GDLoggingParameters(object):
    """
    Contains all necessary settings for the gradient descent algorithm to know how to log the progress.
    
    log_progress:           Whether or not to log the training
    num_cost_evaluations:   How many data points of the cost evaluations should be gathered
    cost_eval_subset_size:  The size of the subset on which the cost is evaluated
    log_file_name:          The name of the file in which the log is saved
    accuracy_func:          The function which evaluates the accuracy of the current model
    prediction_threshold:   If a threshold > 0 is specified, all the values exceeding that threshold will be set to 1
                            while all other values are set to 0. If no threshold is specified, the max value is set
                            to 1 while all others are set to 0.
    """
    log_progress = True
    num_cost_evaluations = 50
    cost_eval_use_subset = True
    cost_eval_subset_size = 5000
    log_file_name = 'gd_log_'

    accuracy_func = None
    prediction_threshold = 0.0

    # do not modify directly
    accuracy_monitors = []
    num_accuracy_monitors = 0

    def __init__(self):
        self.log_file_name = 'gd_log_' + time.strftime("%Y%m%d-%H%M%S")

    def add_accuracy_monitor(self, X, y, subset_size=-1, name=''):
        if len(name) == 0:
            # if no name is specified, add a generic name
            name = 'Data set ' + str(self.num_accuracy_monitors)

        if subset_size > 0:
            m, n = X.shape
            idx = np.random.permutation(m)

            # add the monitor
            self.accuracy_monitors.append({
                'X': X[idx[0:min(subset_size, m)], :],
                'y': y[idx[0:min(subset_size, m)], :],
                'idx': self.num_accuracy_monitors,
                'name': name
            })

            # increase the total number of monitors
            self.num_accuracy_monitors += 1
        else:
            # add the monitor
            self.accuracy_monitors.append({
                'X': X,
                'y': y,
                'idx': self.num_accuracy_monitors,
                'name': name
            })

            # increase the total number of monitors
            self.num_accuracy_monitors += 1
