from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import io
import os
import fairness

app = Flask(__name__)
app.secret_key = 'cs310'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    session['uploaded_csv_file_path'] = file_path

    df = pd.read_csv(file_path)
    table = df.head().to_html(classes="table table-striped", index=False)
    columns = df.columns.tolist()

    return jsonify({'table': table, 'columns': columns})

@app.route('/sensitive-attributes', methods=['POST'])
def sensitive_attributes():
    data = request.get_json()
    return jsonify(success=True)

@app.route('/column-values', methods=['POST'])
def column_values():
    file_path = session.get('uploaded_csv_file_path')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({'values': []}), 400
    
    df = pd.read_csv(file_path)
    data = request.get_json()
    column = data['column']
    unique_values = df[column].unique().tolist()
    
    return jsonify({'values': unique_values})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    tables = []
    file_path = session.get('uploaded_csv_file_path')
    if not file_path or not os.path.exists(file_path):
        return "Error: No CSV file uploaded or file not found.", 400
    df = pd.read_csv(file_path)
    sensitive_attributes = request.form.get('sensitiveAttributes')
    outcome_column = request.form.get('outcomeColumn')
    positive_outcome = request.form.get('positiveOutcome')
    for attribute in sensitive_attributes.split(","):
        tables.append(fairness.outcome_summary(df, attribute, outcome_column, positive_outcome).to_html(classes="table table-striped", index=True).replace('<th>', '<th style="text-align: left;">'))
    return render_template('evaluate.html', tables=tables, outcome_column=outcome_column, positive_outcome=positive_outcome, sensitive_attributes=sensitive_attributes)

if __name__ == '__main__':
    app.run(debug=True)