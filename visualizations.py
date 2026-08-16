import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

def scatter_plot(df: pd.DataFrame, x_column: str, y_column: str):

    if x_column not in df.columns or y_column not in df.columns:
        return None

    if not pd.api.types.is_numeric_dtype(df[x_column]):
        return None

    if not pd.api.types.is_numeric_dtype(df[y_column]):
        return None

    data = df[[x_column, y_column]].dropna()

    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(7,4))

    ax.scatter(
        data[x_column],
        data[y_column]
    )

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"{x_column} vs {y_column}")

    return fig

def histogram_plot(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return None

    if not pd.api.types.is_any_real_numeric_dtype(df[column]):
        return None

    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(df[column].dropna(), bins=30)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {column}")

    return fig

def bar_plot(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return None

    if not pd.api.types.is_object_dtype(df[column]) and not pd.api.types.is_categorical_dtype(df[column]):
        return None

    value_counts = df[column].value_counts(dropna=False)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(value_counts.index.astype(str), value_counts.values)

    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {column}")

    plt.xticks(rotation= 45)

    return fig

def box_plot(df:pd.DataFrame, column:str):
    if column not in df.columns:
        return None

    if not pd.api.types.is_any_real_numeric_dtype(df[column]):
        return None

    data = df[column].dropna()

    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(7,4))
    ax.boxplot(data)
    ax.set_ylabel(column)
    ax.set_title(f"Outlier Detection for {column}")

    return fig

def correlation_heatmap(df: pd.DataFrame):
    numerical_columns= df.select_dtypes(include=np.number).columns.tolist()
        
    if len(numerical_columns) < 2:
        return None

    correlation_matrix = df[numerical_columns].corr()

    fig, ax = plt.subplots(figsize=(7,4))
    image = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.columns)))

    ax.set_xticklabels(correlation_matrix.columns, rotation= 45, ha='right')
    ax.set_yticklabels(correlation_matrix.columns)

    ax.set_title("Correlation Heatmap")
    fig.colorbar(image, ax=ax)

    return fig
