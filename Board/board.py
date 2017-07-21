import numpy as np
import Board.log_data_manager as LogDM

from flask import Flask
from flask import render_template
from flask import url_for

from os import path as os_path
from os import stat as os_stat

app = Flask(__name__)


# ========== Routes ==========

@app.route("/")
@app.route("/home")
def dashboard():
    return render_template('dashboard.html')


@app.route("/nets")
def nets():
    neural_nets = LogDM.get_neural_networks()

    return render_template('nets.html', neural_nets=neural_nets)


@app.route("/nets/<net_id>")
def net_info(net_id):
    neural_net = LogDM.get_net_info(net_id)

    return render_template('net_info.html', neural_net=neural_net, net_id=net_id)


@app.route("/nets/<net_id>/<session_id>")
def net_training_info(net_id, session_id):
    neural_net = LogDM.get_net_training_info(net_id, session_id)

    return render_template('net_training_info.html', neural_net=neural_net, net_id=net_id, session_id=session_id)

# ========== end of routes ==========


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


# we do this override so we can add a "random" value to the link of the css file
# so that the browser is forced to get the latest version of the file and doesn't use any
# cached versions (needed for development when we edit the css and want to see the effects)
def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os_path.join(app.root_path, endpoint, filename)
            values['q'] = int(os_stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == "__main__":
    app.run(debug=True)
