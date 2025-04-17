import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover
from sklearn.naive_bayes import GaussianNB
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


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
    totals = totals.tolist()
    return [sensitive_attr, x_axis, positive, negative, disparate_impact, totals]

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

def actual_vs_predicted_summary(df, sensitive_attr, outcome, positive_value, predictions):
    actual_summary = df.groupby(sensitive_attr)[outcome].value_counts().unstack(fill_value=0)
    actual_totals = actual_summary.sum(axis=1)
    actual_percentages = actual_summary.div(actual_totals, axis=0) * 100
    predicted_summary = df.groupby(sensitive_attr)[predictions].value_counts().unstack(fill_value=0)
    predicted_totals = predicted_summary.sum(axis=1)
    predicted_percentages = predicted_summary.div(predicted_totals, axis=0) * 100
    try:
        actual_percentages.columns = actual_percentages.columns.astype(int)
        predicted_percentages.columns = predicted_percentages.columns.astype(int)
    except:
        pass
    actual_percentages.columns = actual_percentages.columns.astype(str)
    predicted_percentages.columns = predicted_percentages.columns.astype(str)
    actual_positive = actual_percentages[positive_value].round(1).tolist()
    actual_negative = [100 - p for p in actual_positive]
    predicted_positive = predicted_percentages[positive_value].round(1).tolist()
    predicted_negative = [100 - p for p in predicted_positive]
    x_axis = []
    positive_rates = []
    negative_rates = []
    for group, a_pos, a_neg, p_pos, p_neg in zip(actual_percentages.index, actual_positive, actual_negative, predicted_positive, predicted_negative):
        x_axis.append(group)
        x_axis.append(f"{group} (predicted)")
        positive_rates.append(a_pos)
        positive_rates.append(p_pos)
        negative_rates.append(a_neg)
        negative_rates.append(p_neg)
    return [sensitive_attr, x_axis, positive_rates, negative_rates]

def postprocessing_comparison(df, adjusted_df, sensitive_attr, outcome, positive_value, predictions):
    df = df.astype(str)
    adjusted_df = adjusted_df.astype(str)
    x_axis = []
    positive_rates = []
    negative_rates = []
    false_positive_rates = []
    false_negative_rates = []
    prediction_accuracies = []
    for group in df[sensitive_attr].unique():
        original_group_data = df[df[sensitive_attr] == group]
        adjusted_group_data = adjusted_df[adjusted_df[sensitive_attr] == group]
        for i, group_data in enumerate([original_group_data, adjusted_group_data]):
            true_positives = group_data[(group_data[outcome] == positive_value) & (group_data[predictions] == positive_value)]
            true_negatives = group_data[(group_data[outcome] != positive_value) & (group_data[predictions] != positive_value)]
            false_positives = group_data[(group_data[outcome] != positive_value) & (group_data[predictions] == positive_value)]
            false_negatives = group_data[(group_data[outcome] == positive_value) & (group_data[predictions] != positive_value)]
            fpr = (len(false_positives) / (len(false_positives) + len(true_negatives))) * 100 if (len(false_positives) + len(true_negatives)) > 0 else 0
            fnr = (len(false_negatives) / (len(true_positives) + len(false_negatives))) * 100 if (len(true_positives) + len(false_negatives)) > 0 else 0
            total_predictions = len(true_positives) + len(true_negatives) + len(false_positives) + len(false_negatives)
            correct_predictions = len(true_positives) + len(true_negatives)
            pa = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            positive_rate = (len(true_positives) + len(false_positives)) * 100 / total_predictions
            negative_rate = (len(true_negatives) + len(false_negatives)) * 100 / total_predictions
            if i == 0:
                x_axis.append(f"{group} (original)")
            else:
                x_axis.append(f"{group} (transformed)")
            false_positive_rates.append(round(fpr, 1))
            false_negative_rates.append(round(fnr, 1))
            prediction_accuracies.append(round(pa, 1))
            positive_rates.append(round(positive_rate, 1))
            negative_rates.append(round(negative_rate, 1))
    return [x_axis, positive_rates, negative_rates, false_positive_rates, false_negative_rates, prediction_accuracies]
            
@timer
def apply_di_removal(df, outcome, positive_value, sensitive_attr):
    outcome_values = df[outcome].unique()
    negative_value = [str(x) for x in outcome_values if str(x) != str(positive_value)][0]
    label_mapping = {positive_value: 1, negative_value: 0}
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    df[outcome] = df[outcome].map(label_mapping)
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoding_maps = {}
    for column in categorical_columns:
        df[column] = pd.Categorical(df[column])
        encoding_maps[column] = dict(enumerate(df[column].cat.categories))
        df[column] = df[column].cat.codes
    dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df,
        label_names=[outcome],
        protected_attribute_names=[sensitive_attr]
    )
    di_remover = DisparateImpactRemover(repair_level=1.0, sensitive_attribute=sensitive_attr)
    transformed_dataset = di_remover.fit_transform(dataset)
    transformed_df = transformed_dataset.convert_to_dataframe()[0]
    for col, column_map in encoding_maps.items():
        if col in transformed_df.columns:
            transformed_df[col] = transformed_df[col].map(column_map)
    transformed_df[outcome] = transformed_df[outcome].map(reverse_label_mapping)
    transformed_df.to_csv("transformed_output.csv", index=False)

@timer
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
    transformed_df.to_csv("resampled_output.csv", index=False)

def bayes_subgroup_ranker(df, sensitive_column, outcome_column, positive_class_value):
    df = df.astype(str)  
    original_columns = df.columns.tolist()
    X = df.drop(columns=[sensitive_column, outcome_column])
    y = df[outcome_column]
    X = pd.get_dummies(X, drop_first=True)
    model = GaussianNB()
    model.fit(X, y)
    probas = model.predict_proba(X)[:, 1]  
    df['probability'] = probas
    rankings = {}
    sensitive_values = df[sensitive_column].astype(str).unique()
    outcome_values = df[outcome_column].astype(str).unique()
    for sensitive_value in sensitive_values:
        for outcome_value in outcome_values:
            subgroup = df[(df[sensitive_column] == sensitive_value) & (df[outcome_column] == outcome_value)]
            if outcome_value == positive_class_value:  
                subgroup_sorted = subgroup.sort_values(by='probability', ascending=True)  
            else:  
                subgroup_sorted = subgroup.sort_values(by='probability', ascending=False)  
            subgroup_label = (sensitive_value, 'positive' if outcome_value == positive_class_value else 'negative')
            rankings[subgroup_label] = subgroup_sorted[original_columns + ['probability']]  
    return rankings

@timer
def apply_preferential_resampling(df, outcome, positive_value, sensitive_attr):
    df = df.astype(str)
    groups = df[sensitive_attr].unique()
    positive_rate = len(df[df[outcome] == positive_value]) / len(df)
    negative_rate = len(df[df[outcome] != positive_value]) / len(df)
    resampled_subgroups = {}
    expected_sizes = {}
    transformed_df = pd.DataFrame(columns=df.columns)
    for group in groups:
        expected_sizes[(group, "positive")] = int(len(df[df[sensitive_attr] == group]) * positive_rate)
        expected_sizes[(group, "negative")] = int(len(df[df[sensitive_attr] == group]) * negative_rate)
    rankings = bayes_subgroup_ranker(df, sensitive_attr, outcome, positive_value)
    for group in groups:
        for subgroup in ["positive", "negative"]:
            subgroup_df = df[(df[sensitive_attr] == group) & (df[outcome] == positive_value if subgroup == "positive" else df[outcome] != positive_value)]
            subgroup_size = len(subgroup_df)
            expected_size = expected_sizes[(group, subgroup)]
            if subgroup_size < expected_size:
                subgroup_label = (group, "positive" if subgroup == "positive" else "negative")
                subgroup_sorted = rankings[subgroup_label]                
                additional_rows_needed = expected_size - subgroup_size
                to_add = subgroup_sorted.head(additional_rows_needed)
                resampled_subgroups[(group, subgroup)] = pd.concat([subgroup_df, to_add], ignore_index=True)
            elif subgroup_size > expected_size:
                subgroup_label = (group, "positive" if subgroup == "positive" else "negative")
                subgroup_sorted = rankings[subgroup_label]                
                excess_rows = subgroup_size - expected_size
                to_remove = subgroup_sorted.head(excess_rows)
                resampled_subgroups[(group, subgroup)] = subgroup_df.drop(to_remove.index)        
        for subgroup in ["positive", "negative"]:
            transformed_df = pd.concat([transformed_df, resampled_subgroups[(group, subgroup)]], ignore_index=True)  
    transformed_df = transformed_df.drop(columns=['probability'])  
    transformed_df.to_csv("resampled_output.csv", index=False)

@timer
def apply_postprocessing(df, outcome_column, predictions_column, positive_value, sensitive_column, alpha):
    df = df.copy().astype(str)
    negative_value = [x for x in df[outcome_column].unique() if x != positive_value][0]
    rankings = bayes_subgroup_ranker(df, sensitive_column, outcome_column, positive_value)
    overall_positive_rate = (df[predictions_column] == positive_value).mean()
    adjusted_predictions = df[predictions_column].copy()
    for group in df[sensitive_column].unique():
        group_indices = df[df[sensitive_column] == group].index
        group_preds = adjusted_predictions.loc[group_indices]
        group_true = df.loc[group_indices, outcome_column]
        actual_positive_rate = (group_preds == positive_value).mean()
        subgroup_true_rate = (group_true == positive_value).mean()
        target_positive_rate = alpha * overall_positive_rate + (1 - alpha) * subgroup_true_rate
        total = len(group_indices)
        target_positives = round(target_positive_rate * total)
        current_positives = (group_preds == positive_value).sum()
        flips_needed = abs(target_positives - current_positives)
        if flips_needed == 0:
            continue  
        if target_positive_rate > actual_positive_rate:
            candidates = rankings.get((group, 'negative'), pd.DataFrame())
            flipped = 0
            for idx in candidates.index:
                if flipped >= flips_needed:
                    break
                if adjusted_predictions[idx] == negative_value:
                    adjusted_predictions[idx] = positive_value
                    flipped += 1
        elif target_positive_rate < actual_positive_rate:
            candidates = rankings.get((group, 'positive'), pd.DataFrame())
            flipped = 0
            for idx in candidates.index:
                if flipped >= flips_needed:
                    break
                if adjusted_predictions[idx] == positive_value:
                    adjusted_predictions[idx] = negative_value
                    flipped += 1
    result_df = df.copy()
    result_df[predictions_column] = adjusted_predictions
    result_df.drop(columns=['probability'], errors='ignore', inplace=True)
    result_df.to_csv("adjusted_predictions.csv", index=False)