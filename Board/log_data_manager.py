import pickle
import numpy as np

from definitions import ROOT_DIR
from os import listdir as os_listdir
from os import path as os_path

from Utils.anomaly_detector import AnomalyDetector as AnomalyDetector


def get_neural_networks():
    """
    Finds all the neural networks in Logs/NeuralNets
    
    :return: A list of all the neural networks
    """
    neural_nets = []

    path = os_path.join(ROOT_DIR, 'Logs/NeuralNets')

    # loop through all the log files in the log's neural net directory
    if os_path.exists(path):
        for log_file_name in os_listdir(path):
            if log_file_name.endswith('.log'):
                file_path = os_path.join(path, log_file_name)

                with open(file_path, 'rb') as log_file:
                    log_data = pickle.load(log_file)

                    # add the network to the list if it's not yet in it
                    if log_data['network_info'] not in neural_nets:
                        neural_nets.append(log_data['network_info'])

                    continue

    return neural_nets


def get_net_info(net_id: str):
    """
    Gathers relevant information about the specified neural network, such as trainings performed on that network,
    input data added to the network (only if the input was manually added to the log) and relvant network architecture
    information.
    
    :param net_id:  The id of the network whose information should be gathered
    :return:        A dictionary containing the relevant information
    """
    neural_net = {'trainings': [], 'network_info': [], 'input_data': []}

    path = os_path.join(ROOT_DIR, 'Logs/NeuralNets')

    # loop through all the log files in the log's neural net directory
    if os_path.exists(path):
        for log_file_name in os_listdir(path):
            # only consider log files
            if log_file_name.endswith('.log'):
                file_path = os_path.join(path, log_file_name)

                # open the log file
                with open(file_path, 'rb') as log_file:
                    # load the data that is saved in this log file
                    log_data = pickle.load(log_file)

                    # only process the network if it is the right one
                    if log_data['network_info']['id'] == net_id:
                        # only add the neural net's info to the dict if we haven't done so yet
                        if len(neural_net['network_info']) == 0:
                            neural_net['network_info'] = log_data['network_info']

                            for input_data in log_data['input_data']:
                                neural_net['input_data'].append(input_data)

                            # get an approximation of the layer sizes (relative to each other) so we can
                            # display an approximation of the network on the board
                            neural_net['network_info']['layer_sizes'] = get_net_layer_sizes_approximation(neural_net['network_info'])

                        # add all the training sessions
                        for training in log_data['training_sessions']:
                            neural_net['trainings'].append({'session_id': training, 'data': log_data['training_sessions'][training]})

    return neural_net


def get_net_layer_sizes_approximation(network_info: list):
    """
    Takes a neural network info list as input and calculates an approximation of the layer sizes so we can display
    the layer sizes approximation on the board (because the actual network layers usually have too much units to display
    nicely on a page)
    
    :param network_info:    The network info list
    :return:                The approximation of the layer sizes
    """
    layers = network_info['layers']
    # the amount of units to be displayed on the board when approximating the shape of the NN
    layer_sizes = [0] * len(layers)
    # used to calculate the amount of units to be displayed
    layer_ratios = []

    max_units = layers[0]['units']
    max_units_index = 0

    # calculate the ratios between the layers and find the layer with the most units
    for i in range(1, len(layers)):
        layer_ratios.append((layers[i]['units'] * 1.0) / layers[i - 1]['units'])

        if max_units < layers[i]['units']:
            max_units = layers[i]['units']
            max_units_index = i

    # max amount of units to be displayed on the board
    layer_sizes[max_units_index] = min(max_units, 7)

    # calculate the number of units for every layer left to the layer with the most units
    # based on the previously calculated ratios
    for i in range(max_units_index - 1, -1, -1):
        layer_sizes[i] = max(1, int(round(layer_sizes[i + 1] * (1.0 / layer_ratios[i]))))

    # calculate the number of units for every layer right to the layer with the most units
    # based on the previously calculated ratios
    for i in range(max_units_index + 1, len(layer_sizes)):
        layer_sizes[i] = max(1, int(round(layer_sizes[i - 1] * layer_ratios[i - 1])))

    return layer_sizes


def get_net_training_info(net_id: str, session_id: str):
    neural_net = {'training': {}, 'network_info': [], 'input_data': []}

    path = os_path.join(ROOT_DIR, 'Logs/NeuralNets')

    # loop through all the log files in the log's neural net directory
    if os_path.exists(path):
        for log_file_name in os_listdir(path):
            if log_file_name.endswith('.log'):
                file_path = os_path.join(path, log_file_name)

                with open(file_path, 'rb') as log_file:
                    log_data = pickle.load(log_file)

                    # only process the network if it is the right one
                    if log_data['network_info']['id'] == net_id:
                        # add the network info if we haven't done so yet
                        if len(neural_net['network_info']) == 0:
                            neural_net['network_info'] = log_data['network_info']

                        # loop through all the trainings
                        for training in log_data['training_sessions']:
                            # if it is the training we seek to get information on, add it
                            if training == session_id:
                                neural_net['training'] = log_data['training_sessions'][training]

                        for input_data in log_data['input_data']:
                            neural_net['input_data'].append(input_data)

                    continue

    if not neural_net['training']:
        return None

    # get the entries and the costs
    entries = neural_net['training']['entries']
    costs = neural_net['training']['costs']

    # get the anomaly free version of the entries and the costs
    neural_net['training']['entries_without_anom'], neural_net['training']['costs_without_anom'] = get_anomaly_free_data(entries, costs)

    return neural_net


def get_anomaly_free_data(entries: list, costs: list):
    """
    Takes a list of entries and outputs and finds all anomalies and removes them.
    
    :param entries: The entries (x-axis)
    :param costs:   The outputs (y-axis)
    :return:        Anomaly free version of the input and output
    """
    entries = np.matrix(entries).T
    costs = np.matrix(costs).T
    # conserve a copy of the original costs so we can display the original costs on the graph
    costs_original = np.matrix.copy(costs)
    # do a logarithmic transform to improve the anomaly detector's predictions
    costs = np.log(costs)

    detector = AnomalyDetector(multivariate=True)

    # prepare the input for the detector
    data_input = np.hstack((entries, costs))

    # train the detector on the data
    detector.train(np.matrix(data_input))

    # set epsilon to a value which performs reasonably well for detecting the anomalies
    detector.epsilon = 0.003

    # prepare a mask so we only return the values which are not anomalies
    idx_mask = np.ones_like(entries, dtype=bool)
    # set the mask value of the anomalies to false
    idx_mask[detector.find_anomalies_indices(data_input)] = False

    # get the masked data
    new_entries = np.array(entries[idx_mask]).ravel()
    new_costs = np.array(costs_original[idx_mask]).ravel()

    new_entries = new_entries.tolist()
    new_costs = new_costs.tolist()

    return new_entries, new_costs