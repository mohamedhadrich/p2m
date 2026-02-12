"""Retraining pipeline & MLflow registry."""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
import mlflow
import mlflow.sklearn

# ── Available models ─────────────────────────────────────────────
MODELS = {
    "classification": {
        "Random Forest": RandomForestClassifier,
        "Logistic Regression": LogisticRegression,
        "Gradient Boosting": GradientBoostingClassifier,
    },
    "regression": {
        "Random Forest": RandomForestRegressor,
        "Linear Regression": LinearRegression,
        "Gradient Boosting": GradientBoostingRegressor,
    },
}


@dataclass
class RetrainResult:
    model: object
    metrics: dict
    model_name: str
    task_type: str
    run_id: str = None


# ── Feature preparation ──────────────────────────────────────────
def prepare_features(
    df: pd.DataFrame, target: str, feature_cols: list
) -> tuple:
    """Return (X, y, target_encoder).

    * One-hot encodes categorical features.
    * Fills missing values (median for numeric, mode for categorical).
    * Encodes target with LabelEncoder when categorical.
    """
    X = df[feature_cols].copy()
    y = df[target].copy() if target and target in df.columns else None

    # Drop rows where target is NaN
    if y is not None:
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

    # Fill missing — numeric: median, categorical: mode
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include=["object", "category"]).columns:
        mode_val = X[col].mode()
        X[col] = X[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "missing")

    # One-hot encode
    X = pd.get_dummies(X, drop_first=True)

    # Encode target
    target_encoder = None
    if y is not None and y.dtype == "object":
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y), index=y.index, name=target)

    return X, y, target_encoder


def align_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Make X_test columns match X_train (add missing, drop extra)."""
    missing = set(X_train.columns) - set(X_test.columns)
    for col in missing:
        X_test[col] = 0
    return X_test[X_train.columns]


# ── Training ─────────────────────────────────────────────────────
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    model_name: str = "Random Forest",
):
    """Instantiate and fit a scikit-learn model."""
    model_class = MODELS[task_type][model_name]

    if model_name == "Logistic Regression":
        model = model_class(max_iter=1000, random_state=42)
    elif model_name in ("Random Forest", "Gradient Boosting"):
        model = model_class(n_estimators=100, random_state=42)
    else:
        model = model_class()

    model.fit(X, y)
    return model


# ── Retrain + MLflow ─────────────────────────────────────────────
def retrain_with_mlflow(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    model_name: str,
    experiment_name: str = "drift-retrain",
) -> RetrainResult:
    """Train, cross-validate, and log the run to MLflow."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"retrain-{model_name}") as run:
        model = train_model(X, y, task_type, model_name)

        scoring = (
            "f1_weighted" if task_type == "classification" else "neg_mean_squared_error"
        )
        n_splits = min(5, max(2, len(y) // 10))
        cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring=scoring)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_metric("cv_score_mean", cv_scores.mean())
        mlflow.log_metric("cv_score_std", cv_scores.std())
        mlflow.sklearn.log_model(model, "model")

        metrics = {
            "CV Score (mean)": round(float(cv_scores.mean()), 4),
            "CV Score (std)": round(float(cv_scores.std()), 4),
        }

        return RetrainResult(
            model=model,
            metrics=metrics,
            model_name=model_name,
            task_type=task_type,
            run_id=run.info.run_id,
        )
