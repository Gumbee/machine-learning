import pickle
import numpy as np

from flask import Flask
from flask import render_template
from flask import url_for

from definitions import ROOT_DIR
from os import listdir as os_listdir
from os import path as os_path
from os import stat as os_stat

from Utils.anomaly_detector import AnomalyDetector as AnomalyDetector


app = Flask(__name__)

# Routes

@app.route("/")
@app.route("/home")
def dashboard():
    return render_template('dashboard.html')


@app.route("/nets")
def nets():
    neural_nets = get_neural_networks()

    return render_template('nets.html', neural_nets=neural_nets)


@app.route("/nets/<net_id>")
def net_info(net_id):
    neural_net = get_net_info(net_id)

    np.set_printoptions(threshold=np.nan)

    return render_template('net_info.html', neural_net=neural_net, net_id=net_id)


@app.route("/nets/<net_id>/<session_id>")
def net_training_info(net_id, session_id):
    neural_net = get_net_training_info(net_id, session_id)

    np.set_printoptions(threshold=np.nan)

    return render_template('net_training_info.html', neural_net=neural_net, net_id=net_id, session_id=session_id)

# Util Functions

def get_neural_networks():
    neural_nets = []

    path = os_path.join(ROOT_DIR, 'Logs/NeuralNets')

    if os_path.exists(path):
        for log_file_name in os_listdir(path):
            if log_file_name.endswith('.log'):
                file_path = os_path.join(path, log_file_name)

                with open(file_path, 'rb') as log_file:
                    log_data = pickle.load(log_file)

                    if log_data['network_info'] not in neural_nets:
                        neural_nets.append(log_data['network_info'])

                    continue

    return neural_nets


def get_net_info(net_id: str):
    neural_net = {'trainings': [], 'network_info': []}

    path = os_path.join(ROOT_DIR, 'Logs/NeuralNets')

    if os_path.exists(path):
        for log_file_name in os_listdir(path):
            if log_file_name.endswith('.log'):
                file_path = os_path.join(path, log_file_name)

                with open(file_path, 'rb') as log_file:
                    log_data = pickle.load(log_file)

                    if log_data['network_info']['id'] == net_id:
                        if len(neural_net['network_info']) == 0:
                            neural_net['network_info'] = log_data['network_info']

                            layers = neural_net['network_info']['layers']
                            # the amount of units to be displayed on the board when approximating the shape of the NN
                            layer_sizes = [0]*len(layers)
                            # used to calculate the amount of units to be displayed
                            layer_ratios = []

                            max_units = layers[0]['units']
                            max_units_index = 0

                            # calculate the ratios between the layers and find the layer with the most units
                            for i in range(1, len(layers)):
                                layer_ratios.append((layers[i]['units']*1.0)/layers[i-1]['units'])

                                if max_units < layers[i]['units']:
                                    max_units = layers[i]['units']
                                    max_units_index = i

                            # max amount of units to be displayed on the board
                            layer_sizes[max_units_index] = min(max_units, 7)

                            # calculate the number of units for every layer left to the layer with the most units
                            # based on the previously calculated ratios
                            for i in range(max_units_index-1, -1, -1):
                                layer_sizes[i] = max(1, int(round(layer_sizes[i+1]*(1.0/layer_ratios[i]))))

                            # calculate the number of units for every layer right to the layer with the most units
                            # based on the previously calculated ratios
                            for i in range(max_units_index+1, len(layer_sizes)):
                                layer_sizes[i] = max(1, int(round(layer_sizes[i-1]*layer_ratios[i-1])))

                            neural_net['network_info']['layer_sizes'] = layer_sizes

                        for training in log_data['training_sessions']:
                            print(training)
                            neural_net['trainings'].append({'session_id': training, 'data': log_data['training_sessions'][training]})

                    continue

    return neural_net


def get_net_training_info(net_id: str, session_id: str):
    neural_net = {'training': {}, 'network_info': []}

    path = os_path.join(ROOT_DIR, 'Logs/NeuralNets')

    if os_path.exists(path):
        for log_file_name in os_listdir(path):
            if log_file_name.endswith('.log'):
                file_path = os_path.join(path, log_file_name)

                with open(file_path, 'rb') as log_file:
                    log_data = pickle.load(log_file)

                    if log_data['network_info']['id'] == net_id:
                        if len(neural_net['network_info']) == 0:
                            neural_net['network_info'] = log_data['network_info']

                        for training in log_data['training_sessions']:
                            if training == session_id:
                                neural_net['training'] = log_data['training_sessions'][training]

                    continue

    np.set_printoptions(threshold=np.nan)

    entries = neural_net['training']['entries']
    costs = neural_net['training']['costs']

    neural_net['training']['entries_no_anom'], neural_net['training']['costs_no_anom'] = get_anomaly_free_data(entries, costs)

    return neural_net


def get_anomaly_free_data(entries: list, costs: list):
    entries = np.matrix(entries).T
    costs = np.matrix(costs).T
    costs_old = np.matrix.copy(costs)
    costs = np.log(costs)

    detector = AnomalyDetector(multivariate=True)

    x = np.hstack((entries, costs))

    detector.train(x)

    detector.epsilon = 0.003

    idx = np.ones_like(entries, dtype=bool)
    idx[detector.find_anomalies_indices(x)] = False

    new_entries = np.array(entries[idx]).ravel()
    new_costs = np.array(costs_old[idx]).ravel()

    new_entries = new_entries.tolist()
    new_costs = new_costs.tolist()

    return new_entries, new_costs


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os_path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os_stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == "__main__":
    app.run(debug=True)
