"""Business cost estimation for model drift and retraining decisions."""

from dataclasses import dataclass
from utils.performance import PerformanceResult


@dataclass
class CostEstimate:
    """Encapsulates cost breakdown."""
    drift_cost: float  # Cost due to performance degradation
    retrain_cost: float  # Cost of retraining (approximate)
    net_benefit: float  # benefit of retraining = drift_cost - retrain_cost
    recommendation: str  # "Retrain recommended" or "No retrain needed"


def estimate_drift_cost(
    ref_perf: PerformanceResult,
    curr_perf: PerformanceResult,
    n_events: int,
    cost_per_error: float = 100.0,
    task_type: str = "classification",
) -> float:
    """
    Estimate the business cost (in USD or €) of performance degradation.
    
    Args:
        ref_perf: Reference (baseline) model performance.
        curr_perf: Current model performance.
        n_events: Number of predictions/events in the period.
        cost_per_error: Cost per misclassification (default 100, arbitrary units).
        task_type: "classification" or "regression".
    
    Returns:
        Estimated cost (float).
    
    Example:
        If Accuracy drops from 95% to 90% and we have 10,000 predictions,
        that's 500 additional errors → cost = 500 * 100 = 50,000 units.
    """
    
    if task_type == "classification":
        # For classification, assume "cost per error" from accuracy drop
        ref_error_rate = 1 - ref_perf.metrics.get("Accuracy", 0.5)
        curr_error_rate = 1 - curr_perf.metrics.get("Accuracy", 0.5)
    else:
        # For regression, approximate via RMSE change (more errors = worse)
        ref_rmse = ref_perf.metrics.get("RMSE", 1.0)
        curr_rmse = curr_perf.metrics.get("RMSE", 1.0)
        # Assume normalized RMSE in [0, 1]; rough proxy for error rate
        ref_error_rate = min(ref_rmse / 10, 1.0)  # Arbitrary scaling
        curr_error_rate = min(curr_rmse / 10, 1.0)
    
    # Additional errors due to degradation
    additional_errors = (curr_error_rate - ref_error_rate) * n_events
    
    # Only count positive additional errors (actual degradation)
    if additional_errors < 0:
        additional_errors = 0
    
    drift_cost = additional_errors * cost_per_error
    return float(drift_cost)


def estimate_retrain_cost(
    model_name: str,
    n_samples: int,
    hourly_cost: float = 50.0,
    human_review_hours: float = 2.0,
) -> float:
    """
    Estimate the cost of retraining and deploying a new model.
    
    Cost varies based on:
    - n_samples: larger datasets require more compute time (scales with data size).
    - model_name: complex models (e.g., Gradient Boosting, Random Forest) are more expensive than simple ones.
    
    Args:
        model_name: Type of model (e.g., "Random Forest", "Logistic Regression", "Gradient Boosting").
        n_samples: Number of training samples.
        hourly_cost: Cost per compute hour (default 50, arbitrary units).
        human_review_hours: Human time for review/validation (default 2).
    
    Returns:
        Total retrain + validation cost (float).
    """
    
    # Model complexity multiplier
    model_complexity = {
        "Logistic Regression": 0.5,      # cheap, linear
        "Linear Regression": 0.5,
        "Random Forest": 1.5,            # moderate, ensemble
        "Gradient Boosting": 2.0,        # expensive, sequential
        "GradientBoostingClassifier": 2.0,
        "GradientBoostingRegressor": 2.0,
    }
    
    complexity_factor = model_complexity.get(model_name, 1.0)
    
    # Estimate compute hours based on sample size and model complexity
    # Baseline: 0.1 hours for 1000 samples with complexity 1.0
    baseline_hours = 0.1
    compute_hours = baseline_hours * (n_samples / 1000) * complexity_factor
    
    # Compute cost
    compute_cost = compute_hours * hourly_cost
    
    # Human review cost (fixed)
    human_cost = human_review_hours * hourly_cost
    
    total = compute_cost + human_cost
    return float(total)


def make_retrain_decision(
    drift_cost: float,
    retrain_cost: float,
    threshold_ratio: float = 1.5,
) -> CostEstimate:
    """
    Make a retrain decision based on cost‑benefit analysis.
    
    Args:
        drift_cost: Estimated cost of keeping old model (from drift).
        retrain_cost: Estimated cost of retraining.
        threshold_ratio: Multiplier; retrain if drift_cost > retrain_cost * threshold_ratio.
    
    Returns:
        CostEstimate with recommendation.
    
    Logic:
        If drift_cost > threshold_ratio * retrain_cost, it's worth retraining.
        This means the cost of NOT retraining exceeds the cost of retraining.
    """
    net_benefit = drift_cost - retrain_cost
    
    if drift_cost > threshold_ratio * retrain_cost:
        recommendation = "✅ Retrain recommended (drift cost >> retrain cost)"
    elif drift_cost > retrain_cost:
        recommendation = "⚠️ Retrain borderline (drift cost > retrain cost, but marginal)"
    else:
        recommendation = "❌ No retrain needed (drift cost < retrain cost)"
    
    return CostEstimate(
        drift_cost=drift_cost,
        retrain_cost=retrain_cost,
        net_benefit=net_benefit,
        recommendation=recommendation,
    )
