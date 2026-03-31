"""Retraining pipeline & MLflow registry."""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from dataclasses import dataclass
import mlflow
import mlflow.sklearn

# ── LSTM wrapper for sklearn compatibility ──────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    class LSTMRegressor:
        """LSTM wrapper for regression tasks."""
        def __init__(self, units=32, epochs=20, batch_size=32, verbose=0):
            self.units = units
            self.epochs = epochs
            self.batch_size = batch_size
            self.verbose = verbose
            self.model = None
            self.scaler = StandardScaler()

        def fit(self, X, y):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            # Reshape for LSTM: (samples, timesteps, features)
            if len(X.shape) == 2:
                X = X.reshape((X.shape[0], 1, X.shape[1]))

            # Build LSTM model
            self.model = keras.Sequential([
                layers.LSTM(self.units, activation='relu', input_shape=(X.shape[1], X.shape[2])),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mse')
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                          verbose=self.verbose)
            return self

        def predict(self, X):
            X = np.asarray(X, dtype=np.float32)
            if len(X.shape) == 2:
                X = X.reshape((X.shape[0], 1, X.shape[1]))
            return self.model.predict(X, verbose=0).flatten()

    class LSTMClassifier:
        """LSTM wrapper for classification tasks."""
        def __init__(self, units=32, epochs=20, batch_size=32, verbose=0):
            self.units = units
            self.epochs = epochs
            self.batch_size = batch_size
            self.verbose = verbose
            self.model = None
            self.classes_ = None
            self.n_classes_ = None

        def fit(self, X, y):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y)

            # Handle label encoding
            if y.dtype == object or y.dtype.kind in ['U', 'S']:
                le = LabelEncoder()
                y = le.fit_transform(y)
                self.classes_ = le.classes_
            else:
                self.classes_ = np.unique(y)

            self.n_classes_ = len(self.classes_)
            y_encoded = y if self.n_classes_ == 2 else keras.utils.to_categorical(y, self.n_classes_)

            # Reshape for LSTM: (samples, timesteps, features)
            if len(X.shape) == 2:
                X = X.reshape((X.shape[0], 1, X.shape[1]))

            # Build LSTM model
            if self.n_classes_ == 2:
                self.model = keras.Sequential([
                    layers.LSTM(self.units, activation='relu', input_shape=(X.shape[1], X.shape[2])),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(1, activation='sigmoid')
                ])
                self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            else:
                self.model = keras.Sequential([
                    layers.LSTM(self.units, activation='relu', input_shape=(X.shape[1], X.shape[2])),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(self.n_classes_, activation='softmax')
                ])
                self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            self.model.fit(X, y_encoded, epochs=self.epochs, batch_size=self.batch_size,
                          verbose=self.verbose)
            return self

        def predict(self, X):
            X = np.asarray(X, dtype=np.float32)
            if len(X.shape) == 2:
                X = X.reshape((X.shape[0], 1, X.shape[1]))

            probs = self.model.predict(X, verbose=0)
            if self.n_classes_ == 2:
                return self.classes_[(probs > 0.5).astype(int).flatten()]
            else:
                return self.classes_[np.argmax(probs, axis=1)]

        def predict_proba(self, X):
            X = np.asarray(X, dtype=np.float32)
            if len(X.shape) == 2:
                X = X.reshape((X.shape[0], 1, X.shape[1]))

            probs = self.model.predict(X, verbose=0)
            if self.n_classes_ == 2:
                probs_binary = np.column_stack([1 - probs, probs])
                return probs_binary
            else:
                return probs

    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# ── Available models with default parameters ──────────────────────
MODELS = {
    "classification": {
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "Gradient Boosting": GradientBoostingClassifier,
        "SVM": SVC,
        "K-Nearest Neighbors": KNeighborsClassifier,
        "Decision Tree": DecisionTreeClassifier,
        "Naive Bayes": GaussianNB,
        "AdaBoost": AdaBoostClassifier,
    },
    "regression": {
        "Linear Regression": LinearRegression,
        "Random Forest": RandomForestRegressor,
        "Gradient Boosting": GradientBoostingRegressor,
        "SVM": SVR,
        "K-Nearest Neighbors": KNeighborsRegressor,
        "Decision Tree": DecisionTreeRegressor,
        "AdaBoost": AdaBoostRegressor,
    },
}

# Add LSTM if TensorFlow is available
if LSTM_AVAILABLE:
    MODELS["classification"]["LSTM"] = LSTMClassifier
    MODELS["regression"]["LSTM"] = LSTMRegressor

# ── Default hyperparameters per model ──────────────────────────────
DEFAULT_PARAMS = {
    "Logistic Regression": {"max_iter": 1000, "random_state": 42, "C": 1.0},
    "Random Forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1},
    "Gradient Boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "random_state": 42},
    "SVM": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "random_state": 42},
    "K-Nearest Neighbors": {"n_neighbors": 5, "weights": "uniform"},
    "Decision Tree": {"max_depth": 10, "min_samples_split": 5, "random_state": 42},
    "Naive Bayes": {},
    "AdaBoost": {"n_estimators": 50, "learning_rate": 1.0, "random_state": 42},
    "Linear Regression": {},
    "LSTM": {"units": 32, "epochs": 20, "batch_size": 32, "verbose": 0},
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
    custom_params: dict = None,
    scale_features: bool = True,
):
    """Instantiate and fit a scikit-learn model with optional custom parameters.

    Args:
        X: Feature matrix
        y: Target vector
        task_type: 'classification' or 'regression'
        model_name: Name of model to use
        custom_params: Dictionary of custom hyperparameters (overrides defaults)
        scale_features: Whether to scale features (important for SVM, KNN)
    """
    model_class = MODELS[task_type][model_name]

    # Get default parameters
    params = DEFAULT_PARAMS.get(model_name, {}).copy()

    # Override with custom parameters if provided
    if custom_params:
        params.update(custom_params)

    # Create model instance
    model = model_class(**params)

    # Scale features for models that benefit from it
    X_train = X.copy()
    if scale_features and model_name in ["SVM", "K-Nearest Neighbors"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X)
        model.scaler = scaler  # Store scaler for later use

    # Fit model
    model.fit(X_train, y)
    return model


# ── Retrain + MLflow ─────────────────────────────────────────────
def retrain_with_mlflow(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    model_name: str,
    custom_params: dict = None,
    experiment_name: str = "drift-retrain",
) -> RetrainResult:
    """Train, cross-validate, and log the run to MLflow.

    Args:
        X: Feature matrix
        y: Target vector
        task_type: 'classification' or 'regression'
        model_name: Name of model to use
        custom_params: Dictionary of custom hyperparameters
        experiment_name: MLflow experiment name
    """
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"retrain-{model_name}") as run:
        model = train_model(X, y, task_type, model_name, custom_params=custom_params)

        scoring = (
            "f1_weighted" if task_type == "classification" else "neg_mean_squared_error"
        )
        n_splits = min(5, max(2, len(y) // 10))

        # Use KFold with fixed random_state for reproducible CV splits
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", X.shape[0])

        # Log custom parameters if provided
        if custom_params:
            for param_name, param_value in custom_params.items():
                mlflow.log_param(f"custom_{param_name}", param_value)

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


# ── Helper functions for parameter management ───────────────────
def get_default_params(model_name: str) -> dict:
    """Get default hyperparameters for a model."""
    return DEFAULT_PARAMS.get(model_name, {}).copy()


def get_parameter_info(model_name: str) -> dict:
    """Get information about parameters for a specific model.

    Returns:
        Dictionary with parameter names and their default values/types
    """
    defaults = get_default_params(model_name)

    # Define parameter info (type hints for UI)
    param_info = {
        # Logistic Regression
        "C": ("float", 0.1, 10.0),
        "max_iter": ("int", 100, 5000),

        # Random Forest
        "n_estimators": ("int", 10, 500),
        "max_depth": ("int", 2, 30),

        # Gradient Boosting
        "learning_rate": ("float", 0.01, 1.0),

        # SVM
        "kernel": ("select", ["linear", "rbf", "poly"]),
        "gamma": ("select", ["scale", "auto"]),

        # K-Nearest Neighbors
        "n_neighbors": ("int", 1, 50),
        "weights": ("select", ["uniform", "distance"]),

        # Decision Tree
        "min_samples_split": ("int", 2, 20),

        # AdaBoost

        # LSTM
        "units": ("int", 16, 128),
        "epochs": ("int", 10, 50),
        "batch_size": ("select", [16, 32, 64, 128]),
    }

    return defaults, param_info


# ── Model and parameter optimization ────────────────────────────
from dataclasses import dataclass

@dataclass
class ModelRecommendation:
    """Recommendation for best model and parameters."""
    model_name: str
    params: dict
    cv_score: float
    cv_std: float
    training_time: float


def find_best_model_and_params(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    cv_folds: int = 3,
    quick_mode: bool = True,
) -> list:
    """Find best model and hyperparameters using grid search.

    Args:
        X: Feature matrix
        y: Target vector
        task_type: 'classification' or 'regression'
        cv_folds: Number of cross-validation folds
        quick_mode: If True, test fewer parameter combinations

    Returns:
        List of ModelRecommendation objects sorted by performance
    """
    import time
    from sklearn.model_selection import cross_validate

    results = []
    model_options = MODELS[task_type]

    # Define parameter grids for each model
    param_grids = {
        "Logistic Regression": [
            {"C": 0.1, "max_iter": 1000},
            {"C": 1.0, "max_iter": 1000},
            {"C": 10.0, "max_iter": 1000},
        ] if not quick_mode else [
            {"C": 1.0, "max_iter": 1000},
        ],

        "Random Forest": [
            {"n_estimators": 50, "max_depth": 5},
            {"n_estimators": 100, "max_depth": 10},
            {"n_estimators": 200, "max_depth": 15},
        ] if not quick_mode else [
            {"n_estimators": 100, "max_depth": 10},
        ],

        "Gradient Boosting": [
            {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
        ] if not quick_mode else [
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
        ],

        "SVM": [
            {"kernel": "rbf", "C": 1.0},
            {"kernel": "linear", "C": 1.0},
        ] if not quick_mode else [
            {"kernel": "rbf", "C": 1.0},
        ],

        "K-Nearest Neighbors": [
            {"n_neighbors": 3, "weights": "uniform"},
            {"n_neighbors": 5, "weights": "distance"},
        ] if not quick_mode else [
            {"n_neighbors": 5, "weights": "uniform"},
        ],

        "Decision Tree": [
            {"max_depth": 5, "min_samples_split": 5},
            {"max_depth": 10, "min_samples_split": 2},
        ] if not quick_mode else [
            {"max_depth": 10, "min_samples_split": 5},
        ],

        "Naive Bayes": [{}],

        "AdaBoost": [
            {"n_estimators": 50, "learning_rate": 1.0},
            {"n_estimators": 100, "learning_rate": 0.5},
        ] if not quick_mode else [
            {"n_estimators": 50, "learning_rate": 1.0},
        ],

        "Linear Regression": [{}],

        "LSTM": [
            {"units": 32, "epochs": 15, "batch_size": 32},
            {"units": 64, "epochs": 20, "batch_size": 16},
        ] if not quick_mode else [
            {"units": 32, "epochs": 10, "batch_size": 32},
        ] if LSTM_AVAILABLE else [],
    }

    scoring = "f1_weighted" if task_type == "classification" else "neg_mean_squared_error"

    # Test each model with different parameter combinations
    for model_name, model_class in model_options.items():
        param_combinations = param_grids.get(model_name, [{}])

        for params in param_combinations:
            try:
                # Get full params with defaults
                full_params = get_default_params(model_name).copy()
                full_params.update(params)

                # Create and train model
                start_time = time.time()
                model = model_class(**full_params)

                # Apply scaling if needed
                X_train = X.copy()
                if model_name in ["SVM", "K-Nearest Neighbors"]:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X)

                # Handle LSTM separately (custom cross-validation)
                if model_name == "LSTM" and LSTM_AVAILABLE:
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    fold_scores = []

                    for train_idx, test_idx in kf.split(X_train):
                        X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
                        y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

                        lstm_fold = model_class(**full_params)
                        lstm_fold.fit(X_fold_train, y_fold_train)

                        if task_type == "classification":
                            fold_score = lstm_fold.model.evaluate(
                                np.asarray(X_fold_test, dtype=np.float32).reshape(
                                    X_fold_test.shape[0], 1, X_fold_test.shape[1]
                                ),
                                y_fold_test, verbose=0
                            )[1]
                        else:
                            fold_score = lstm_fold.model.evaluate(
                                np.asarray(X_fold_test, dtype=np.float32).reshape(
                                    X_fold_test.shape[0], 1, X_fold_test.shape[1]
                                ),
                                y_fold_test, verbose=0
                            )
                            fold_score = -fold_score

                        fold_scores.append(fold_score)

                    training_time = time.time() - start_time
                    cv_score = np.mean(fold_scores)
                    cv_std = np.std(fold_scores)
                else:
                    # Cross-validate for sklearn models with fixed random_state
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    cv_results = cross_validate(
                        model, X_train, y, cv=kf, scoring=scoring,
                        return_train_score=False
                    )

                    training_time = time.time() - start_time
                    cv_score = cv_results['test_score'].mean()
                    cv_std = cv_results['test_score'].std()

                # Store result
                recommendation = ModelRecommendation(
                    model_name=model_name,
                    params=params,
                    cv_score=cv_score,
                    cv_std=cv_std,
                    training_time=training_time
                )
                results.append(recommendation)

            except Exception as e:
                print(f"Error testing {model_name} with {params}: {e}")
                continue

    # Sort by score (higher is better for f1, higher is better for negative MSE)
    results.sort(key=lambda x: x.cv_score, reverse=True)

    return results


def format_recommendation_for_display(recommendation: ModelRecommendation, task_type: str) -> dict:
    """Format recommendation for display in Streamlit.

    Returns:
        Dictionary with formatted display info
    """
    metric_name = "F1 Score" if task_type == "classification" else "MSE"

    return {
        "Model": recommendation.model_name,
        "Parameters": str(recommendation.params),
        metric_name: f"{recommendation.cv_score:.4f}",
        "Std Dev": f"{recommendation.cv_std:.4f}",
        "Training Time (s)": f"{recommendation.training_time:.2f}",
    }