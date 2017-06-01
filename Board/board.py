import pickle

from flask import Flask
from flask import render_template
from flask import url_for

from definitions import ROOT_DIR
from os import listdir as os_listdir
from os import path as os_path
from os import stat as os_stat

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
