from ast import Continue
import pandas as pd 
import numpy as np
import io


def analyze_missing_values(df: pd.DataFrame) -> dict:
    missing_values= df.isnull().sum()
    missing_percentage= (missing_values / len(df))*100

    missing_info = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
    })

    missing_info = missing_info[
        missing_info['Missing Count'] > 0
    ]

    missing_info = missing_info.sort_values(
        by= ['Missing Percentage'],
        ascending=False
    )

    remove_threshold = 70
    impute_threshold = 1

    columns_to_remove= []
    columns_to_impute= []
    columns_with_few_missing= []

    if not missing_info.empty:
        columns_to_remove = missing_info[missing_info['Missing Percentage'] > remove_threshold].index.tolist()
        columns_to_impute = missing_info[(missing_info['Missing Percentage']> impute_threshold) & (missing_info['Missing Percentage'] <= remove_threshold)].index.tolist()

        # Identify columns that were not flagged for removal or imputation
        all_cols_with_missing = missing_info.index.tolist()
        columns_with_few_missing= [col for col in df.columns if col not in all_cols_with_missing]

    else:
        columns_with_few_missing= df.columns.tolist()

    return{
        "missing_info_df": missing_info,
        "columns_to_remove": columns_to_remove,
        "columns_to_impute": columns_to_impute,
        "columns_with_few_missing": columns_with_few_missing,
        "has_missing_values": not missing_info.empty,
        "remove_threshold": remove_threshold,
        "impute_threshold": impute_threshold
    }

def analyze_duplicates(df: pd.DataFrame) -> dict:
    if df.empty:
        return{
            "duplicate_count": 0,
            "duplicate_percentage": 0.0,
            "has_duplicates": False,
            "duplicate_rows_df": pd.DataFrame()
        }

    duplicate_count= df.duplicated().sum()
    total_rows= len(df)

    duplicate_percentage= (duplicate_count / total_rows)*100 if total_rows > 0 else 0.0
    duplicate_rows_df= df[df.duplicated(keep=False)]

    return{
        "duplicate_count": duplicate_count,
        "duplicate_percentage": round(duplicate_percentage, 2),
        "has_duplicates": duplicate_count > 0,
        "duplicate_rows_df": duplicate_rows_df
    }

def analyze_column_types(df: pd.DataFrame) -> dict:

    column_types = {}

    for col in df.columns:

        # 1. Numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numerical"
            continue

        # 2. Boolean columns
        if pd.api.types.is_bool_dtype(df[col]):
            column_types[col] = "categorical"
            continue

        # 3. Object / categorical columns
        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col])
        ):

            non_null_values = df[col].dropna()

            # Empty column
            if non_null_values.empty:
                column_types[col] = "categorical"
                continue

            # Only try date detection if the column
            # actually looks like a date column
            sample_values = non_null_values.astype(str).head(20)

            looks_like_date = sample_values.str.contains(
                r"[-/]|:",
                regex=True
            ).mean() > 0.5

            if looks_like_date:

                datetime_series = pd.to_datetime(
                    non_null_values,
                    errors="coerce",
                    format="mixed"
                )

                valid_ratio = datetime_series.notna().mean()

                if valid_ratio > 0.8:
                    column_types[col] = "date"
                else:
                    column_types[col] = "categorical"

            else:
                column_types[col] = "categorical"

            continue

        # 4. Other datatypes
        column_types[col] = "other"

    return column_types

def numerical_analysis(df: pd.DataFrame) -> dict:
    numerical_stats = {}
    numerical_columns= df.select_dtypes(include=np.number).columns.tolist()

    for col in numerical_columns:
        column_data = df[col]

        # Calculate Statistics
        count = column_data.count()
        null_count = column_data.isnull().sum()
        mean = column_data.mean()
        median= column_data.median()
        std_dev = column_data.std()
        min_val = column_data.min() 
        max_val = column_data.max()

        # Store the statistics for this column
        numerical_stats[col] = {
            'count': count,
            'null_count': null_count,
            'mean': mean,
            'median': median,
            'std_dev': std_dev,
            'min': min_val,
            'max': max_val
        }
    
    return numerical_stats

def outlier_detection(df: pd.DataFrame) -> dict:
    outlier_data = {}

    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

    if not numerical_columns:
        print("No numerical columns found in the DataFrame.")
        return outlier_data

    for col in numerical_columns:
        column_data_no_nan = df[col].dropna()

        if column_data_no_nan.empty:
            continue

        Q1 = column_data_no_nan.quantile(0.25)
        Q3 = column_data_no_nan.quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)].tolist()

        if outliers:
            outlier_data[col] = outliers

    return outlier_data

def categorical_analysis(df: pd.DataFrame) -> dict:
    categorical_analysis ={}
    categorical_cols = df.select_dtypes(include=['object']).columns

    if not categorical_cols.any():
        print("No categorical columns found in the DataFrame.")
        return categorical_analysis

    for col in categorical_cols:
        col_data= df[col]

        # Calculate value counts, including NaN if present
        value_counts = col_data.value_counts(dropna=False)

        # Calculate Percentage
        total_count = value_counts.sum()
        category_percentages = (value_counts / total_count) *100

        # Get uniques categories (index of value_counts)
        unique_categories = value_counts.index.tolist()

        most_frequent_category = value_counts.index[0] if not value_counts.empty else None
        most_frequent_count = value_counts.iloc[0] if not value_counts.empty else 0

        categorical_analysis[col] = {
            'unique_categories': unique_categories,
            'category_counts': value_counts.to_dict(),
            'category_percentages': category_percentages.to_dict(),
            'most_frequent_category': most_frequent_category,
            'most_frequent_count': most_frequent_count
        }
    
    return categorical_analysis

def correlation_analysis(df: pd.DataFrame) -> dict:
    
    numerical_columns= df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numerical_columns) < 2:
        return{
            "correlation_matrix": pd.DataFrame(),
            "strong_relationships": []
        }

    correlation_matrix = df[numerical_columns]
    correlation_matrix = correlation_matrix.corr()
    strong_relationships= []
    
    threshold= 0.7
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):

            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]

            correlation_value = correlation_matrix.loc[col1, col2]

            if abs(correlation_value) >= threshold:
                strong_relationships.append({
                    "column_1": col1,
                    "column_2": col2,
                    "correlation": round(correlation_value, 2)
                })

    return{
        "correlation_matrix": correlation_matrix,
        "strong_relationships": strong_relationships
    }

