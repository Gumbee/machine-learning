from os import makedirs as os_makedirs
from os import path as os_path

import numpy as np
import pickle
import uuid

from definitions import ROOT_DIR
from parameters import GDLoggingParameters as GDLoggingParameters
from parameters import GradientDescentParameters as GradientDescentParameters


class LogHandler(object):
    """
    class that handles logging.
    
    """

    def __init__(self, gd_log_parameters: GDLoggingParameters = None):
        self.gd_log_parameters = gd_log_parameters or GDLoggingParameters()
        self.log_dict = {'training_sessions': {}}

    def log_gd_progress(self, session_id: str, epoch_num: int, batch_num: int, init_theta: np.matrix, X: np.matrix, y: np.matrix, gd_parameters: GradientDescentParameters):
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
                cost = cost_func(init_theta, X[cst_idx, :], y[cst_idx, :], reg_lambda, **func_args)
            else:
                cost = cost_func(init_theta, X, y, reg_lambda, **func_args)

            # get relative change of the cost function
            self.log_dict['training_sessions'][session_id]['rel_chng'] = cost - self.log_dict['training_sessions'][session_id]['prev_cst']
            # update previous cost to current cost
            self.log_dict['training_sessions'][session_id]['prev_cst'] = cost
            # update the entry number
            self.log_dict['training_sessions'][session_id]['entry_num'] += 1

            entry_num: int = self.log_dict['training_sessions'][session_id]['entry_num']
            rel_chng: int = self.log_dict['training_sessions'][session_id]['rel_chng']

            # log progress
            self.print_table_entry(entry_num, epoch_num + 1, cost, rel_chng, 1.0)
            self.add_gd_entry(session_id, entry_num, epoch_num + 1, cost, rel_chng)

    def open_gd_session(self, initial_error: float):
        # only print the table if we want to log the progress
        if self.gd_log_parameters.log_progress:
            self.print_table_header('P', 'EP', 'COST', 'CHNG', 'ASCL')
            self.print_table_entry(0, 1, initial_error, initial_error, 1.00)

            session_id = uuid.uuid4().hex
            self.log_dict['training_sessions'][session_id] = {'entries': [], 'epochs': [], 'costs': [], 'rel_chngs': [], 'entry_num': 0, 'rel_chng': 0, 'prev_cst': 0}
            self.add_gd_entry(session_id, 0, 1, initial_error, initial_error)

            return session_id
        else:
            return -1

    def close_gd_session(self):
        self.write_gd_progress_to_file()

    def add_gd_entry(self, session_id: str, entry_num: int, epoch_num: int, cost: float, rel_chng: float):
        self.log_dict['training_sessions'][session_id]['entries'].append(entry_num)
        self.log_dict['training_sessions'][session_id]['epochs'].append(epoch_num)
        self.log_dict['training_sessions'][session_id]['costs'].append(cost)
        self.log_dict['training_sessions'][session_id]['rel_chngs'].append(rel_chng)

    def register_network(self, net_id: str, name: str, model):
        self.log_dict['network_info'] = {}
        self.log_dict['network_info']['id'] = net_id
        self.log_dict['network_info']['name'] = name
        self.log_dict['network_info']['layers'] = [self.get_layer_info(layer) for layer in model['layers']]
        self.write_gd_progress_to_file()

    @staticmethod
    def get_layer_info(layer):
        return {'units': layer.units, 'has_bias': layer.has_bias, 'activation': layer.activation_name}

    def write_gd_progress_to_file(self):
        path = os_path.join(ROOT_DIR, 'Logs/' + self.gd_log_parameters.log_file_name + '.log')

        if not os_path.exists(os_path.dirname(path)):
            os_makedirs(os_path.dirname(path))

        np.set_printoptions(threshold=np.nan)

        with open(path, 'wb') as log_file:
            pickle.dump(self.log_dict, log_file)

    # ================= Util Functions =================

    @staticmethod
    def print_table_header(First: str, Second: str, Third: str, Fourth: str, Fifth: str):
        print('\n\033[91m', '{:>4s}'.format(str(First)), '{:>1s}'.format('|'), '{:>5s}'.format(str(Second)),
              '{:>1s}'.format('|'),
              '{:>15s}'.format(str(Third)), '{:>1s}'.format('|'), '{:>15s}'.format(str(Fourth)),
              '{:>1s}'.format('|'),
              '{:>10s}'.format(str(Fifth)), '{:>1s}'.format('|'), '\033[0m')
        print('\033[91m', '{:─>63s}'.format(''), '\033[0m')

    @staticmethod
    def print_table_entry(First: int, Second: int, Third: float, Fourth: float, Fifth: float):
        print('\033[91m', '{:>4d}'.format(First), '{:1s}'.format('|'), '{:>5d}'.format(Second),
              '{:>1s}'.format('|'),
              '{:>15.6e}'.format(Third), '{:>1s}'.format('|'), '{:>15.6e}'.format(Fourth), '{:>1s}'.format('|'),
              '{:>10.3f}'.format(Fifth), '{:>1s}'.format('|'), '\033[0m')
