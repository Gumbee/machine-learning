import pickle

from flask import Flask
from flask import render_template

from definitions import ROOT_DIR
from os import listdir as os_listdir
from os import path as os_path

app = Flask(__name__)


@app.route("/")
def dashboard():
    log_dicts = []

    path = os_path.join(ROOT_DIR, 'Logs/')

    if os_path.exists(path):
        for log_file_name in os_listdir(path):
            if log_file_name.endswith('.log'):
                file_path = os_path.join(path, log_file_name)

                with open(file_path, 'rb') as log_file:
                    log_dicts.append(pickle.load(log_file))
                    continue

    return render_template('dashboard.html', log=log_dicts)

if __name__ == "__main__":
    app.run()

