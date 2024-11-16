from flask import Flask, render_template, request, jsonify, session, flash
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
        for i in range(len(resultset[4])):
            if resultset[4][i] < 0.8:
                favoured_group_index = resultset[4].index(max(resultset[4])) 
                favoured_group = resultset[1][favoured_group_index]
                flash(f"Group {resultset[1][i]} disparately impacted")
                fairness.apply_di_removal(df, outcome_column, positive_outcome, attribute, favoured_group)
        resultsets.append(resultset)
    return render_template('evaluate.html', resultsets=resultsets, outcome_column=outcome_column, positive_outcome=positive_outcome, sensitive_attributes=sensitive_attributes)

@app.route('/evaluatepred', methods=['POST'])
def evaluatepred():
    resultsets = []
    file_path = session.get('uploaded_csv_file_path')
    if not file_path or not os.path.exists(file_path):
        return "Error: No CSV file uploaded or file not found.", 400
    df = pd.read_csv(file_path)
    sensitive_attributes = request.form.get('sensitiveAttributes')
    outcome_column = request.form.get('outcomeColumn')
    positive_outcome = request.form.get('positiveOutcome')
    prediction_column = request.form.get('predictionsColumn')
    for attribute in sensitive_attributes.split(","):
        resultset = fairness.predicted_outcome_summary(df, attribute, outcome_column, positive_outcome, prediction_column)
        resultsets.append(resultset)
    return render_template('evaluatepred.html', resultsets=resultsets, outcome_column=outcome_column, positive_outcome=positive_outcome, sensitive_attributes=sensitive_attributes, prediction_column=prediction_column)

if __name__ == '__main__':
    app.run(debug=True)