import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover

def outcome_summary(df, sensitive_attr, outcome, positive_value):
    summary = df.groupby(sensitive_attr)[outcome].value_counts().unstack(fill_value=0)
    totals = summary.sum(axis=1)
    summary_percentages = summary.div(totals, axis=0) * 100
    try:
        summary_percentages.columns = summary_percentages.columns.astype(int)
    except:
        pass
    summary_percentages.columns = summary_percentages.columns.astype(str)
    largest_positive_rate = summary_percentages[positive_value].max()
    summary_percentages['Disparate Impact Ratio'] = summary_percentages[positive_value] / largest_positive_rate
    x_axis = summary_percentages.index.tolist()
    positive = summary_percentages[positive_value].round(1).tolist()
    negative = [100 - i for i in positive]
    disparate_impact = summary_percentages['Disparate Impact Ratio'].round(3).tolist()
    return [sensitive_attr, x_axis, positive, negative, disparate_impact]

def predicted_outcome_summary(df, sensitive_attr, outcome, positive_value, predictions):
    x_axis = []  
    false_positive_rates = []
    false_negative_rates = [] 
    prediction_accuracies = []  
    df = df.astype(str)
    for group in df[sensitive_attr].unique():
        group_data = df[df[sensitive_attr] == group]
        true_positives = group_data[(group_data[outcome] == positive_value) & (group_data[predictions] == positive_value)]
        true_negatives = group_data[(group_data[outcome] != positive_value) & (group_data[predictions] != positive_value)]
        false_positives = group_data[(group_data[outcome] != positive_value) & (group_data[predictions] == positive_value)]
        false_negatives = group_data[(group_data[outcome] == positive_value) & (group_data[predictions] != positive_value)]
        fpr = (len(false_positives) / (len(false_positives) + len(true_negatives))) * 100 if (len(false_positives) + len(true_negatives)) > 0 else 0
        fnr = (len(false_negatives) / (len(true_positives) + len(false_negatives))) * 100 if (len(true_positives) + len(false_negatives)) > 0 else 0
        total_predictions = len(true_positives) + len(true_negatives) + len(false_positives) + len(false_negatives)
        correct_predictions = len(true_positives) + len(true_negatives)
        pa = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        x_axis.append(group)
        false_positive_rates.append(round(fpr, 1))
        false_negative_rates.append(round(fnr, 1))
        prediction_accuracies.append(round(pa, 1))
    return [sensitive_attr, x_axis, false_positive_rates, false_negative_rates, prediction_accuracies]

def apply_di_removal(df, outcome, positive_value, sensitive_attr):
    outcome_values = df[outcome].unique()
    negative_value = [str(x) for x in outcome_values if str(x) != str(positive_value)][0]
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoding_maps = {}
    for column in categorical_columns:
        df[column] = pd.Categorical(df[column])
        encoding_maps[column] = dict(enumerate(df[column].cat.categories))
        df[column] = df[column].cat.codes
    dataset = BinaryLabelDataset(
        favorable_label=positive_value,
        unfavorable_label=negative_value,
        df=df,
        label_names=[outcome],
        protected_attribute_names=[sensitive_attr]
    )
    di_remover = DisparateImpactRemover(repair_level=1.0, sensitive_attribute=sensitive_attr)
    transformed_dataset = di_remover.fit_transform(dataset)
    transformed_df = transformed_dataset.convert_to_dataframe()[0]
    categorical_transformed_df = transformed_df.copy()
    for col, column_map in encoding_maps.items():
        if col in categorical_transformed_df.columns:
            categorical_transformed_df[col] = categorical_transformed_df[col].map(column_map)
    categorical_transformed_df.to_csv("transformed_output.csv", index=False)

def apply_resampling(df, outcome, positive_value, sensitive_attr):
    df = df.astype(str)
    groups = df[sensitive_attr].unique()
    positive_rate = len(df[df[outcome] == positive_value]) / len(df)
    negative_rate = len(df[df[outcome] != positive_value]) / len(df)
    resampled_subgroups = {}
    expected_sizes = {}
    transformed_df = pd.DataFrame(columns=df.columns)
    for group in groups:
        stringdf = df.astype(str)
        resampled_subgroups[(group, "positive")] = df[(df[sensitive_attr] == group) & (df[outcome] == positive_value)]
        resampled_subgroups[(group, "negative")] = df[(df[sensitive_attr] == group) & (df[outcome] != positive_value)]
        expected_sizes[(group, "positive")] = int(len(df[df[sensitive_attr] == group]) * positive_rate)
        expected_sizes[(group, "negative")] = int(len(df[df[sensitive_attr] == group]) * negative_rate)

        for subgroup in ["positive", "negative"]:
            if expected_sizes[(group, subgroup)] > len(resampled_subgroups[(group, subgroup)]):
                duplicated_rows = pd.DataFrame(columns=df.columns)
                for i in range(expected_sizes[(group, subgroup)] - len(resampled_subgroups[(group, subgroup)])):
                    duplicated_rows = pd.concat([duplicated_rows, resampled_subgroups[(group, subgroup)].sample(n=1)], ignore_index=True)
                resampled_subgroups[(group, subgroup)] = pd.concat([resampled_subgroups[(group, subgroup)], duplicated_rows], ignore_index=True)
            elif expected_sizes[(group, subgroup)] < len(resampled_subgroups[(group, subgroup)]):
                for i in range(len(resampled_subgroups[(group, subgroup)]) - expected_sizes[(group, subgroup)]):
                    resampled_subgroups[(group, subgroup)].drop(resampled_subgroups[(group, subgroup)].sample(n=1).index, inplace=True)

            transformed_df = pd.concat([transformed_df, resampled_subgroups[(group, subgroup)]], ignore_index=True)
   
    print(expected_sizes)
    transformed_df.to_csv("resampled_output.csv", index=False)