from os import makedirs as os_makedirs
from os import path as os_path

import Utils.feature_manager as FM
import numpy as np
import pickle
import uuid

from definitions import ROOT_DIR
from parameters import GDLoggingParameters as GDLoggingParameters
from parameters import GradientDescentParameters as GradientDescentParameters


class LogHandler(object):
    """
    Class that handles logging.
    
    Features:
        - visualize multidimensional data sets
        - monitor the model's loss over the training set
        - monitor the model's accuracy over a certain data set
    """

    def __init__(self, gd_log_parameters: GDLoggingParameters = None):
        self.gd_log_parameters = gd_log_parameters or GDLoggingParameters()
        self.log_dict = {'training_sessions': {}, 'input_data': []}
        self.eigenvectors = None

    def log_gd_progress(self, session_id: str, epoch_num: int, batch_num: int, current_theta: np.matrix, X: np.matrix, y: np.matrix, gd_parameters: GradientDescentParameters):
        """
        Logs a step of the gradient descent algorithm.
        
        :param session_id:      current session's number
        :param epoch_num:       current epoch number
        :param batch_num:       current batch number
        :param current_theta:   current weights
        :param X:               the data set on which the model is being trained
        :param y:               the data set's expected output
        :param gd_parameters:   the parameters with which the gradient descent algorithm is applied
        """
        # get relevant gradient descent parameters
        cost_func = gd_parameters.cost_func
        reg_lambda = gd_parameters.reg_lambda
        func_args = gd_parameters.func_args
        batch_size = gd_parameters.batch_size

        # get relevant logging parameters
        log_progress = self.gd_log_parameters.log_progress
        num_cost_eval = self.gd_log_parameters.num_cost_evaluations
        cost_eval_use_subset = self.gd_log_parameters.cost_eval_use_subset
        cost_eval_subset_size = self.gd_log_parameters.cost_eval_subset_size

        m, _ = X.shape

        # only log the progress and reevaluate the cost every other iteration
        if log_progress and batch_num / batch_size % int(max((m / batch_size) / num_cost_eval, 1)) == 0:
            if cost_eval_use_subset and cost_eval_subset_size < m:
                # if we only want to use a subset of the total training set to determine the cost value then
                # we shuffle the indices of the training set so that we determine the cost with random entries
                # of the training set
                cst_idx = np.random.permutation(m)
                cst_idx = cst_idx[0:cost_eval_subset_size]

                # reevaluate cost function
                cost = cost_func(current_theta, X[cst_idx, :], y[cst_idx, :], reg_lambda, **func_args)
            else:
                cost = cost_func(current_theta, X, y, reg_lambda, **func_args)

            # get relative change of the cost function
            self.log_dict['training_sessions'][session_id]['rel_chng'] = cost - self.log_dict['training_sessions'][session_id]['prev_cst']
            # update previous cost to current cost
            self.log_dict['training_sessions'][session_id]['prev_cst'] = cost

            rel_chng: int = self.log_dict['training_sessions'][session_id]['rel_chng']
            entry_num: int = self.log_dict['training_sessions'][session_id]['entry_num']

            # log progress
            self.print_table_entry(entry_num, epoch_num + 1, cost, rel_chng, 1.0)
            self.add_gd_entry(session_id, epoch_num + 1, cost, rel_chng)

            self.log_gd_accuracy(session_id)

    def log_gd_accuracy(self, session_id: str):
        """
        Lets the model predict the output for each data set that is being monitored and compares the predictions
        to the actual expected output. The accuracy per data set is then added to the log.
        
        :param session_id: 
        :return: 
        """
        # only monitor the accuracy if a mean to calculate the accuracy is provided
        if self.gd_log_parameters.accuracy_func is None:
            return

        # get the data sets which we monitor
        monitors = self.gd_log_parameters.accuracy_monitors
        # get the function that will calculate the accuracy
        accuracy_func = self.gd_log_parameters.accuracy_func
        # get the threshold (if provided, every value in the prediction greater than that threshold
        # will be converted to a 1 and all the others will be a 0). If not provided, the max value is converted
        # to 1 and all other values are converted to 0)
        threshold = self.gd_log_parameters.prediction_threshold

        # if we don't have any data sets which we want to monitor, return
        if len(monitors) == 0:
            return

        # iterate over all data sets and add the accuracy to the log
        for data_set in monitors:
            accuracy = accuracy_func(data_set['X'], data_set['y'], threshold=threshold)
            # log progress
            self.log_dict['training_sessions'][session_id]['accuracies'][data_set['idx']].append(accuracy)

    def open_gd_session(self, initial_error: float):
        """
        Opens a logging sessions for the gradient descent algorithm (or any extension of it).
        
        :param initial_error:   the current error (loss)
        :return:                the new session's id
        """
        # only print the table if we want to log the progress
        if self.gd_log_parameters.log_progress:
            # print the header and the initial error
            self.print_table_header('P', 'EP', 'COST', 'CHNG', 'ASCL')
            self.print_table_entry(0, 1, initial_error, initial_error, 1.00)

            # generate a session id
            session_id = uuid.uuid4().hex

            # define the log dictionary
            self.log_dict['training_sessions'][session_id] = \
            {
                'entries': [],
                'epochs': [],
                'costs': [],
                'accuracies': [],
                'accuracies_names': [],
                'rel_chngs': [],
                'entry_num': 0,
                'rel_chng': 0,
                'prev_cst': 0
            }

            # add the first entry (initial error) to the log
            self.add_gd_entry(session_id, 1, initial_error, initial_error)

            # get the number of data sets whose accuracies we monitor
            num_monitors = len(self.log_dict['training_sessions'][session_id]['accuracies'])

            # if we have some accuracy monitor we haven't added to the log yet, add them
            if num_monitors < self.gd_log_parameters.num_accuracy_monitors:
                for i in range(num_monitors, self.gd_log_parameters.num_accuracy_monitors):
                    self.log_dict['training_sessions'][session_id]['accuracies'].append([])
                    self.log_dict['training_sessions'][session_id]['accuracies_names'].append(self.gd_log_parameters.accuracy_monitors[i]['name'])

            # return the session id
            return session_id
        else:
            return -1

    def close_gd_session(self):
        """
        Closes the logging session and writes the log to a file.
        """
        # write the log to a file
        self.write_gd_progress_to_file()

    def add_gd_entry(self, session_id: str, epoch_num: int, cost: float, rel_chng: float):
        """
        Adds a gradient descent step entry to the log.
        
        :param session_id:  The session id of the session to which we add the entry
        :param epoch_num:   The current epoch's number
        :param cost:        The cost (loss) which we add to the entry
        :param rel_chng:    The relative change of the cost compared to the last entry
        """
        # update the entry number
        self.log_dict['training_sessions'][session_id]['entry_num'] += 1
        entry_num: int = self.log_dict['training_sessions'][session_id]['entry_num']

        # add the values to the log
        self.log_dict['training_sessions'][session_id]['entries'].append(entry_num)
        self.log_dict['training_sessions'][session_id]['epochs'].append(epoch_num)
        self.log_dict['training_sessions'][session_id]['costs'].append(cost)
        self.log_dict['training_sessions'][session_id]['rel_chngs'].append(rel_chng)

    def add_data_set(self, X: np.matrix, classes: np.array, subset_size: int = -1):
        """
        Adds a data set (which can then be visualized on the board) to the log.
        
        :param X:           The data set
        :param classes:     The classes corresponding to the data set (the output values corresponding to the data set)
        :param subset_size: If specified only a subset of the data set will be added to the log 
        """
        # if we haven't calculated any eigenvectors yet, we use the first added data set to calculate
        # the eigenvectors (used to project the data onto fewer dimensions)
        if self.eigenvectors is None:
            _, self.eigenvectors = FM.pca(X, 3)

        # if we have a two dimensional class set we have to process the additional data
        if classes.ndim > 1:
            m, n = classes.shape
            classes_tmp = np.zeros(m)

            # if we only have one value per row, take that as class
            if n == 1:
                classes_tmp = classes.ravel()
            else:
                # iterate over each column and set the class of the example
                # as the index of the column with the highest value
                for i in range(m):
                    classes_tmp[i] = np.where(classes[i] == max(classes[i]))[0]

            classes = classes_tmp

        # if we haved specified a subset size, only add a subset of the data set to the log
        if subset_size > 0:
            # choose the subset randomly
            idx = np.random.permutation(X.shape[0])
            # project the data set onto 3 dimensions
            Z = FM.project_data(X[idx[0:min(subset_size, X.shape[0])], :], self.eigenvectors, 3)
            # get the subset's corresponding classes
            c = classes[idx[0:min(subset_size, X.shape[0])]]
        else:
            # project the whole data set onto 3 dimensions
            Z = FM.project_data(X, self.eigenvectors, 3)
            # get the classes
            c = classes

        # convert the x-axis, y-axis, z-axis and classes to lists
        x = np.ndarray.tolist(Z[:, 0])
        y = np.ndarray.tolist(Z[:, 1])
        z = np.ndarray.tolist(Z[:, 2])
        c = np.ndarray.tolist(c)

        # add the processed data to the log
        self.log_dict['input_data'].append({'x': x, 'y': y, 'z': z, 'c': c})

    def add_accuracy_monitor(self, X, y, subset_size=-1, name=''):
        self.gd_log_parameters.add_accuracy_monitor(X, y, subset_size, name)

    def register_network(self, network):
        """
        Registers a neural network to the log.
        
        :param network: The network
        """
        self.log_dict['network_info'] = {}
        self.log_dict['network_info']['id'] = network.id
        self.log_dict['network_info']['name'] = network.name
        self.log_dict['network_info']['layers'] = [self.get_layer_info(layer) for layer in network.model['layers']]
        self.write_gd_progress_to_file()

    def write_gd_progress_to_file(self):
        """
        Writes the log to a file.
        """
        # get the log folder's path and join it with the file name
        path = os_path.join(ROOT_DIR, 'Logs/' + self.gd_log_parameters.log_file_name + '.log')

        # if the path doesn't exist, create the path
        if not os_path.exists(os_path.dirname(path)):
            os_makedirs(os_path.dirname(path))

        # write the log dictionary to the file
        with open(path, 'wb') as log_file:
            pickle.dump(self.log_dict, log_file)

    # ================= Util Functions =================

    @staticmethod
    def get_layer_info(layer):
        """
        Helper method to extract information from a NeuralLayer and save it in a dictionary.

        :param layer:   the layer whose information is to be extracted
        :return:        a dictionary containing the relevant information about the layer
        """
        return {'units': layer.units, 'has_bias': layer.has_bias, 'activation': layer.activation_name}

    @staticmethod
    def print_table_header(First: str, Second: str, Third: str, Fourth: str, Fifth: str):
        """
        Prints a table header with the specified values.
        
        :param First:   First value that in the header
        :param Second:  Second value that in the header
        :param Third:   Third value that in the header
        :param Fourth:  Fourth value that in the header
        :param Fifth:   Fifth value that in the header
        """
        print('\n\033[91m', '{:>4s}'.format(str(First)), '{:>1s}'.format('|'), '{:>5s}'.format(str(Second)),
              '{:>1s}'.format('|'),
              '{:>15s}'.format(str(Third)), '{:>1s}'.format('|'), '{:>15s}'.format(str(Fourth)),
              '{:>1s}'.format('|'),
              '{:>10s}'.format(str(Fifth)), '{:>1s}'.format('|'), '\033[0m')
        print('\033[91m', '{:─>63s}'.format(''), '\033[0m')

    @staticmethod
    def print_table_entry(First: int, Second: int, Third: float, Fourth: float, Fifth: float):
        """
        Prints a table entry with the specified values.
        
        :param First:   First value that in the header
        :param Second:  Second value that in the header
        :param Third:   Third value that in the header
        :param Fourth:  Fourth value that in the header
        :param Fifth:   Fifth value that in the header
        """
        print('\033[91m', '{:>4d}'.format(First), '{:1s}'.format('|'), '{:>5d}'.format(Second),
              '{:>1s}'.format('|'),
              '{:>15.6e}'.format(Third), '{:>1s}'.format('|'), '{:>15.6e}'.format(Fourth), '{:>1s}'.format('|'),
              '{:>10.3f}'.format(Fifth), '{:>1s}'.format('|'), '\033[0m')
