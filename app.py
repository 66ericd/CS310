from flask import Flask, render_template, request, jsonify, session, send_file, redirect, url_for
import pandas as pd
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
    table = table.replace('<thead>', '<thead><style>th { text-align: left; }</style>')

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
    resultsets = []
    file_path = session.get('uploaded_csv_file_path')
    if not file_path or not os.path.exists(file_path):
        return "Error: No CSV file uploaded or file not found.", 400
    df = pd.read_csv(file_path)
    sensitive_attributes = request.form.get('sensitiveAttributes')
    outcome_column = request.form.get('outcomeColumn')
    positive_outcome = request.form.get('positiveOutcome')
    for attribute in sensitive_attributes.split(","):
        resultset = fairness.outcome_summary(df, attribute, outcome_column, positive_outcome)
        resultsets.append(resultset)
    return render_template('evaluate.html', resultsets=resultsets, outcome_column=outcome_column, positive_outcome=positive_outcome, sensitive_attributes=sensitive_attributes)

@app.route('/removedisparate', methods=['POST'])
def removedisparate():
    file_path = session.get('uploaded_csv_file_path')
    df = pd.read_csv(file_path)
    sensitive_attribute = request.form.get('attribute')
    outcome_column = request.form.get('outcomeColumn')
    positive_outcome = request.form.get('positiveOutcome')
    fairness.apply_di_removal(df, outcome_column, positive_outcome, sensitive_attribute)
    session['transformed_file'] = "transformed_output.csv"
    return redirect(url_for('disparate_impact'))

@app.route('/disparate-impact')
def disparate_impact():
    original_file_path = session.get('uploaded_csv_file_path') 
    original_df = pd.read_csv(original_file_path)
    original_head = original_df.head().to_html(classes="table table-striped", index=False)
    original_head = original_head.replace('<thead>', '<thead><style>th { text-align: left; }</style>')
    transformed_file_path = session.get('transformed_file')    
    transformed_df = pd.read_csv(transformed_file_path)
    transformed_head = transformed_df.head().to_html(classes="table table-striped", index=False)
    transformed_head = transformed_head.replace('<thead>', '<thead><style>th { text-align: left; }</style>')
    return render_template('disparate_impact.html', original_head=original_head, transformed_head=transformed_head)

@app.route('/download-file')
def download_file():
    output_file = session.get('transformed_file')
    return send_file(output_file, as_attachment=True, download_name="transformed_output.csv", mimetype="text/csv")

if __name__ == '__main__':
    app.run(debug=True)
