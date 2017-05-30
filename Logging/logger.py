import time

from os import path as os_path
from os import makedirs as os_makedirs
from definitions import ROOT_DIR


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


class LogHandler(object):
    """
    class that handles logging.
    
    """
    gd_log_parameters = None

    def __init__(self, gd_log_parameters: GDLoggingParameters = None):
        self.gd_log_parameters = gd_log_parameters

    @staticmethod
    def write_gd_progress_to_file(iteration: int, epoch: int, cost: float, rel_chng: float, file_name: str):
        path = os_path.join(ROOT_DIR, 'Logs/' + file_name)

        if not os_path.exists(os_path.dirname(path)):
            os_makedirs(os_path.dirname(path))

        with open(path, 'a') as log_file:
            print(iteration, epoch, cost, rel_chng, file=log_file)
