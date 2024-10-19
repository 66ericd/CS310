import pandas as pd

def outcome_summary(df, sensitive_attr, outcome, positive_value):
    summary = df.groupby(sensitive_attr)[outcome].value_counts().unstack(fill_value=0)
    if positive_value in summary.columns:
        summary.columns = ['Negative', 'Positive']
    else:
        summary.columns = ['Positive', 'Negative']
    totals = summary.sum(axis=1)
    summary_percentages = summary.div(totals, axis=0) * 100
    largest_positive_rate = summary_percentages['Positive'].max()
    summary_percentages['Disparate Impact Ratio'] = summary_percentages['Positive'] / largest_positive_rate
    summary_percentages['Group Size'] = totals
    summary_percentages.index = [f"{index}" for index in summary.index]
    summary_percentages['Positive'] = summary_percentages['Positive'].round(1).astype(str) + '%'
    summary_percentages['Negative'] = summary_percentages['Negative'].round(1).astype(str) + '%'
    summary_percentages['Disparate Impact Ratio'] = summary_percentages['Disparate Impact Ratio'].round(3)
    summary_percentages.index.name = f"{sensitive_attr.capitalize()}"
    return summary_percentages



