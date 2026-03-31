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
    """Heuristic: classification if <=20 unique values or non-numeric."""
    if y.dtype == "object" or y.dtype.name == "category" or y.nunique() <= 20:
        return "classification"
    return "regression"


def evaluate_model(model, X, y, task_type: str = None) -> PerformanceResult:
    """Evaluate a trained model and return metrics."""
    if task_type is None:
        task_type = detect_task_type(y)

    y_pred = model.predict(X)

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
