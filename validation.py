import pandas as pd


def validate_dataframe(df):
    if df.empty:
        return False,"The uploaded dataset is empty."
    if len(df.columns) < 2:
        return False, "Dataset must contain at least 2 columns."
    if df.columns.duplicated().any():
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        return False, f"Duplicate column names found: {duplicate_columns}"
    if df.isnull().all().any():
        empty_columns = df.colmns[df.isnull().all()].tolist()
        return False,f"completely empty columns found: {empty_columns}"

    return True, "Dataset is valid."
