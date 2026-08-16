import pandas as pd
import numpy as np

from sklearn.linear_model import (LogisticRegression, LinearRegression)
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Classification
def train_classification_model(X_train, X_test, y_train, y_test, selected_model = None):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300,random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50,random_state=42)
    }

    if selected_model is not None:
        models = {selected_model: models[selected_model]}

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results[name] = {
            "Accuracy": round(
                accuracy_score(y_test, predictions), 4
            ),
            "Precision": round(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ), 4
            ),
            "Recall": round(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ), 4
            ),
            "F1 Score": round(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ), 4
            )
        }
        trained_models[name] = model

    results_df = pd.DataFrame(results).T.reset_index()
    results_df.rename(columns={"index": "Model"}, inplace=True)


    return results_df, trained_models

# Regression
def train_regression_model(X_train, X_test, y_train, y_test, selected_model = None):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42,n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50,random_state=42)
    }

    if selected_model is not None:
            models = {selected_model: models[selected_model]}
        
    results = []
    trained_models = {}

    for model_name,model in models.items():
        model.fit(X_train,y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test,predictions)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": model_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse,4),
            "R2 Score": round(r2,4)
        })

        # Store Trained model
        trained_models[model_name] = model

    results_df = pd.DataFrame(results)

    return results_df,trained_models

# Classification Evaluation 
def evaluate_classification_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    # Classification Report
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    # ROC-AUC
    roc_auc = None
    try: 
        probabilities = model.predict_proba(X_test)

        #Binary Classification
        if probabilities.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, probabilities[:, 1])


    except Exception:
        pass

    return{
        "predictions": predictions,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": roc_auc
    }

# Regression evaluation
def evaluate_regression_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, predictions)
    residuals = y_test - predictions

    return{
        "predictions": predictions,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "residuals": residuals
    }

# Feature Importance
def get_feature_importance(model, feature_names):
    # Tree-based models
    if hasattr(model, "feature_importances_"):
        importance= np.asarray(model.feature_importances_)

    # Linear Models
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)

        # Binary / multiclass classification
        if len(np.array(coefficients).shape) > 1:
            importance= np.mean(np.abs(coefficients), axis=0)
        else:
            importance= np.abs(coefficients)

    else:
        return None

    # Make sure number of features matches
    if len(feature_names) != len(importance):
        return None

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return importance_df