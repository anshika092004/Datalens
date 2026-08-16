import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split

def create_preprocessing_pipeline(df, selected_features):

    # Select features
    X = df[selected_features].copy()
    feature_names = X.columns.tolist()

    numerical_columns = X.select_dtypes(include= ["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("Scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[(
        "numerical", numerical_pipeline,numerical_columns),  
        ("categorical", categorical_pipeline, categorical_columns)])

    return preprocessor

def prepare_data(df, target_column, selected_features, test_size=0.2, problem_type="Regression"):

    X = df[selected_features].copy()
    y = df[target_column].copy()

    # Save original feature names
    feature_names = X.columns.tolist()
    
    # Remove rows where target is missing
    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(df, selected_features)

    # Train-test split
    if problem_type == "Claffication":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit only on training data
    X_train_processed =preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    return{
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names
    }