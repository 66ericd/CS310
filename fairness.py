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



def apply_di_removal(df, outcome, positive_value, sensitive_attr, favoured_group):
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoding_maps = {}
    for column in categorical_columns:
        df[column] = pd.Categorical(df[column])
        encoding_maps[column] = dict(enumerate(df[column].cat.categories))
        df[column] = df[column].cat.codes
    for column, mapping in encoding_maps[sensitive_attr].items():
        if mapping == favoured_group:
            favoured_group_encoded = column
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
    categorical_transformed_df = transformed_df.copy()
    for col, column_map in encoding_maps.items():
        if col in categorical_transformed_df.columns: 
            categorical_transformed_df[col] = categorical_transformed_df[col].map(column_map)
    categorical_transformed_df.to_csv("transformed_output.csv", index=False)



