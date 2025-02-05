from flask import Flask, render_template, request, jsonify, session, send_file, redirect, url_for
import numpy as np
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

@app.route('/evaluatepred', methods=['POST'])
def evaluatepred():
    resultsets = {}
    file_path = session.get('uploaded_csv_file_path')
    if not file_path or not os.path.exists(file_path):
        return "Error: No CSV file uploaded or file not found.", 400
    df = pd.read_csv(file_path)
    sensitive_attributes = request.form.get('sensitiveAttributes').split(",")
    outcome_column = request.form.get('outcomeColumn')
    positive_outcome = request.form.get('positiveOutcome')
    prediction_column = request.form.get('predictionsColumn')
    for attribute in sensitive_attributes:
        resultsets[attribute] = []
        resultsets[attribute].append(fairness.actual_vs_predicted_summary(df, attribute, outcome_column, positive_outcome, prediction_column))
        resultsets[attribute].append(fairness.predicted_outcome_summary(df, attribute, outcome_column, positive_outcome, prediction_column))
    return render_template('evaluatepred.html', resultsets=resultsets, outcome_column=outcome_column, positive_outcome=positive_outcome, sensitive_attributes=sensitive_attributes, prediction_column=prediction_column)


@app.route('/removedisparate', methods=['POST'])
def removedisparate():
    file_path = session.get('uploaded_csv_file_path')
    df = pd.read_csv(file_path)
    sensitive_attribute = request.form.get('attribute')
    outcome_column = request.form.get('outcomeColumn')
    positive_outcome = request.form.get('positiveOutcome')
    fairness.apply_di_removal(df, outcome_column, positive_outcome, sensitive_attribute)
    session['transformed_file'] = "transformed_output.csv"
    return redirect(url_for('disparate_impact', outcome_column=outcome_column, sensitive_attribute=sensitive_attribute))

@app.route('/disparate-impact')
def disparate_impact():
    original_file_path = session.get('uploaded_csv_file_path') 
    original_df = pd.read_csv(original_file_path)
    transformed_file_path = session.get('transformed_file')    
    transformed_df = pd.read_csv(transformed_file_path)
    sensitive_group_column = request.args.get("sensitive_attribute")
    outcome_column = request.args.get("outcome_column")
    sensitive_groups = original_df[sensitive_group_column].unique()
    resultsets = []
    for column in original_df.columns:
        if column != outcome_column and pd.api.types.is_numeric_dtype(original_df[column]):
            original_values = original_df[column].astype(float)
            transformed_values = transformed_df[column].astype(float)
            if not (original_values == transformed_values).all():
                unique_values = set(original_values).union(set(transformed_values))
                x_axis = list(sorted(unique_values))  
                bin_count = 10  
                bin_edges = np.linspace(min(x_axis), max(x_axis), bin_count + 1)
                bin_labels = [f'[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})' for i in range(len(bin_edges) - 1)]
                original_binned = pd.cut(original_values, bins=bin_edges, labels=bin_labels, include_lowest=True)
                transformed_binned = pd.cut(transformed_values, bins=bin_edges, labels=bin_labels, include_lowest=True)
                original_frequencies = {group: [0] * len(bin_labels) for group in sensitive_groups}
                transformed_frequencies = {group: [0] * len(bin_labels) for group in sensitive_groups}
                for i, group in enumerate(sensitive_groups):
                    group_mask = original_df[sensitive_group_column] == group
                    group_data = original_binned[group_mask]
                    for j, bin_label in enumerate(bin_labels):
                        original_frequencies[group][j] = sum(group_data == bin_label)
                for i, group in enumerate(sensitive_groups):
                    group_mask = transformed_df[sensitive_group_column] == group
                    group_data = transformed_binned[group_mask]
                    for j, bin_label in enumerate(bin_labels):
                        transformed_frequencies[group][j] = sum(group_data == bin_label)
                resultsets.append({
                    'column': column,
                    'x_axis': bin_labels,
                    'original_frequencies': original_frequencies,
                    'transformed_frequencies': transformed_frequencies
                })
    return render_template('disparate_impact.html', resultsets=resultsets)

@app.route('/download-file')
def download_file():
    output_file = session.get('transformed_file')
    return send_file(output_file, as_attachment=True, download_name="transformed_output.csv", mimetype="text/csv")

@app.route('/resampling', methods=['POST'])
def resampling():
    file_path = session.get('uploaded_csv_file_path')
    df = pd.read_csv(file_path)
    sensitive_attribute = request.form.get('attribute')
    outcome_column = request.form.get('outcomeColumn')
    positive_outcome = request.form.get('positiveOutcome')
    fairness.apply_preferential_resampling(df, outcome_column, positive_outcome, sensitive_attribute)
    session['resampled_file'] = "resampled_output.csv"
    return redirect(url_for('resample', outcome_column=outcome_column, sensitive_attribute=sensitive_attribute, positive_outcome=positive_outcome))

@app.route('/resample')
def resample():
    original_file_path = session.get('uploaded_csv_file_path') 
    original_df = pd.read_csv(original_file_path)
    resampled_file_path = session.get('resampled_file')    
    resampled_df = pd.read_csv(resampled_file_path)
    sensitive_group_column = request.args.get("sensitive_attribute")
    outcome_column = request.args.get("outcome_column")
    positive_outcome = request.args.get("positive_outcome")
    original_results = fairness.outcome_summary(original_df, sensitive_group_column, outcome_column, positive_outcome)
    resampled_results = fairness.outcome_summary(resampled_df, sensitive_group_column, outcome_column, positive_outcome)
    return render_template('resample.html', original_results=original_results, resampled_results=resampled_results)

@app.route('/download-file2')
def download_file2():
    output_file = session.get('resampled_file')
    return send_file(output_file, as_attachment=True, download_name="resampled_output.csv", mimetype="text/csv")

@app.route('/postprocessing', methods=['POST'])
def postprocessing():
    file_path = session.get('uploaded_csv_file_path')
    df = pd.read_csv(file_path)
    sensitive_attribute = request.form.get('attribute')
    outcome_column = request.form.get('outcomeColumn')
    prediction_column = request.form.get('predictionColumn')
    positive_outcome = request.form.get('positiveOutcome')
    alpha = float(request.form.get('alphaValue'))
    fairness.apply_postprocessing(df, outcome_column, prediction_column, positive_outcome, sensitive_attribute, alpha)
    session['adjusted_file'] = "adjusted_predictions.csv"
    return redirect(url_for("postprocess", outcome_column=outcome_column, sensitive_attribute=sensitive_attribute, positive_outcome=positive_outcome, prediction_column=prediction_column))

@app.route('/postprocess')
def postprocess():
    original_file_path = session.get('uploaded_csv_file_path')
    original_df = pd.read_csv(original_file_path)
    adjusted_file_path = session.get('adjusted_file')
    adjusted_df = pd.read_csv(adjusted_file_path)
    sensitive_group_column = request.args.get("sensitive_attribute")
    outcome_column = request.args.get("outcome_column")
    positive_outcome = request.args.get("positive_outcome")
    prediction_column = request.args.get("prediction_column")
    postprocessing_results = fairness.postprocessing_comparison(original_df, adjusted_df, sensitive_group_column, outcome_column, positive_outcome, prediction_column)
    return render_template("postprocess.html", postprocessing_results=postprocessing_results)

@app.route('/download-file3')
def download_file3():
    output_file = session.get('adjusted_file')
    return send_file(output_file, as_attachment=True, download_name="adjusted_output.csv", mimetype="text/csv")

if __name__ == '__main__':
    app.run(debug=True)
