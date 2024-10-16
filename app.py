from flask import Flask, render_template, request, jsonify
import pandas as pd
import io

app = Flask(__name__)
df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    file = request.files['file']
    stream = io.StringIO(file.stream.read().decode('UTF-8'))
    df = pd.read_csv(stream)
    table = df.head().to_html(classes="table table-striped", index=False)
    columns = df.columns.tolist()  
    return jsonify({'table': table, 'columns': columns})

@app.route('/sensitive-attributes', methods=['POST'])
def sensitive_attributes():
    data = request.get_json()
    checked_attributes = data['attributes']
    print('Received sensitive attributes:', checked_attributes) 
    return jsonify(success=True)

@app.route('/column-values', methods=['POST'])
def column_values():
    global df
    data = request.get_json()
    column = data['column']
    unique_values = df[column].unique().tolist()
    return jsonify({'values': unique_values})

if __name__ == '__main__':
    app.run(debug=True)
