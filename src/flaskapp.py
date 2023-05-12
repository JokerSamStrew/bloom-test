from flask import Flask, send_from_directory, request, render_template
import os
from bloom560m import Bloom560m

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html', result="")


@app.route("/bloom560m/request", methods=['POST', 'GET'])
def bloom560m_request():
    if request.method != 'POST':
        return render_template('bloom560m.html')

    output = {
        'search_mode' : request.form['search_mode'],
        'result_length' : int(request.form['result_length']),
        'request_string' : request.form['request_string'],
        'result' : ''
    }

    if request.form['request_string'] == '':
        return render_template('bloom560m.html')

    try:
        if output['search_mode'] == "BEAM_SEARCH":
            model = Bloom560m(output['request_string'], output['result_length'])
            output['result'] += model.beam_search()
            print(output['result'])
        elif output['search_mode'] == "GREEDY_SEARCH":
            model = Bloom560m(output['request_string'], output['result_length'])
            output['result'] += model.greedy_search()
            print(output['result'])
        elif output['search_mode'] == "SAMPLING_TOP_SEARCH":
            model = Bloom560m(output['request_string'], output['result_length'])
            output['result'] += model.sampling_top()
            print(output['result'])
        else:
            return render_template('bloom560m.html', res="WRONG INPUT")
        
        output['result'] = output['result'].split('\n')
    except Exception as ex:
        output['request_string'] = str(ex)

    return render_template('bloom560m.html', **output)


if __name__ == "__main__":
    app.run()
