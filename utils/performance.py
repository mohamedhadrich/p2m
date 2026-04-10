"""Model performance evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from dataclasses import dataclass


@dataclass
class PerformanceResult:
    metrics: dict
    task_type: str


def detect_task_type(y: pd.Series) -> str:
    """Infer task type with a conservative heuristic.

    Logic:
    - Non-numeric targets are classification.
    - Numeric targets are classification only when values are integer-like and
      cardinality is small relative to sample size (typical label encoding case).
    - Otherwise treat as regression to avoid misclassifying low-cardinality
      numeric regression targets.
    """
    y_non_null = y.dropna()
    if y_non_null.empty:
        return "classification"

    if y_non_null.dtype == "object" or y_non_null.dtype.name == "category" or y_non_null.dtype == "bool":
        return "classification"

    if pd.api.types.is_numeric_dtype(y_non_null):
        unique_count = y_non_null.nunique()
        n_samples = len(y_non_null)
        # Integer-like numeric targets with low cardinality are often classes.
        is_integer_like = np.allclose(y_non_null, np.round(y_non_null), atol=1e-10)
        relative_cardinality = unique_count / max(1, n_samples)
        if is_integer_like and unique_count <= 15 and relative_cardinality <= 0.2:
            return "classification"

    return "regression"


def evaluate_model(model, X, y, task_type: str = None) -> PerformanceResult:
    """Evaluate a trained model and return metrics."""
    if task_type is None:
        task_type = detect_task_type(y)

    X_eval = X
    # Backward-compatibility: older training flow may store a fitted scaler
    # directly on the model instance instead of using a sklearn Pipeline.
    if hasattr(model, "scaler") and getattr(model, "scaler") is not None:
        X_eval = model.scaler.transform(X)

    y_pred = model.predict(X_eval)

    if task_type == "classification":
        avg = "binary" if len(set(y)) <= 2 else "weighted"
        metrics = {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, average=avg, zero_division=0),
            "Recall": recall_score(y, y_pred, average=avg, zero_division=0),
            "F1 Score": f1_score(y, y_pred, average=avg, zero_division=0),
        }
    else:
        metrics = {
            "MSE": mean_squared_error(y, y_pred),
            "MAE": mean_absolute_error(y, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y, y_pred))),
            "R²": r2_score(y, y_pred),
        }

    return PerformanceResult(metrics=metrics, task_type=task_type)


def check_performance_drop(
    ref_result: PerformanceResult,
    curr_result: PerformanceResult,
    threshold: float = 0.05,
) -> dict:
    """Compare reference vs current metrics and flag degradation.

    Convention: change = (curr - ref) / ref (uniform for all metrics)

    Interpretation:
    - ERROR metrics (RMSE, MAE, MSE): negative change = improvement ✅
    - SCORE metrics (R², Accuracy, F1): positive change = improvement ✅
    """
    drops = {}
    primary_metric_improved = False

    # First pass: check if primary error metric improved
    if "RMSE" in ref_result.metrics:
        ref_rmse = ref_result.metrics["RMSE"]
        curr_rmse = curr_result.metrics["RMSE"]
        rmse_change = (curr_rmse - ref_rmse) / max(abs(ref_rmse), 1e-8)
        # For error metrics: negative = improvement
        if rmse_change < 0:
            primary_metric_improved = True

    # Second pass: calculate all metrics with uniform formula
    for name in ref_result.metrics:
        ref_val = ref_result.metrics[name]
        curr_val = curr_result.metrics[name]

        # Uniform formula for all metrics
        change = (curr_val - ref_val) / max(abs(ref_val), 1e-8)

        # Interpret degradation based on metric type
        if name in ("MSE", "MAE", "RMSE"):
            # Error metrics: negative = improvement, positive = degradation
            degraded = change > threshold
        else:
            # Score metrics: positive = improvement, negative = degradation
            degraded = change < -threshold
            # Ignore R² degradation if RMSE improved (regression robustness)
            if name == "R²" and primary_metric_improved:
                degraded = False

        drops[name] = {
            "ref": round(ref_val, 4),
            "curr": round(curr_val, 4),
            "change": round(change, 4),
            "degraded": degraded,
        }
    return drops


def deployment_readiness(
    old_result: PerformanceResult,
    new_result: PerformanceResult,
    min_improvement: float = 0.0,
) -> dict:
    """Return whether the new model is better enough to deploy.

    Decision rule uses a primary metric by task:
    - classification: prefer F1 Score (fallback Accuracy)
    - regression: prefer RMSE lower-is-better (fallback R² higher-is-better)
    """
    task_type = new_result.task_type or old_result.task_type
    old_metrics = old_result.metrics
    new_metrics = new_result.metrics

    if task_type == "classification":
        primary = "F1 Score" if "F1 Score" in new_metrics else "Accuracy"
        direction = "higher"
    else:
        primary = "RMSE" if "RMSE" in new_metrics else "R²"
        direction = "lower" if primary == "RMSE" else "higher"

    old_val = old_metrics.get(primary)
    new_val = new_metrics.get(primary)

    if old_val is None or new_val is None:
        return {
            "can_deploy": False,
            "primary_metric": primary,
            "direction": direction,
            "old": old_val,
            "new": new_val,
            "delta": None,
            "reason": "Primary metric unavailable for deployment comparison.",
        }

    delta = new_val - old_val
    if direction == "higher":
        improved = delta > min_improvement
    else:
        improved = (-delta) > min_improvement

    if improved:
        reason = (
            f"New model improves {primary}: {old_val:.4f} -> {new_val:.4f}."
        )
    else:
        reason = (
            f"New model does not improve {primary} enough: "
            f"{old_val:.4f} -> {new_val:.4f}."
        )

    return {
        "can_deploy": improved,
        "primary_metric": primary,
        "direction": direction,
        "old": float(old_val),
        "new": float(new_val),
        "delta": float(delta),
        "reason": reason,
    }
