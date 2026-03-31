"""Drift topology analysis: classify and interpret drift patterns."""

from dataclasses import dataclass
from utils.drift import DriftResult
from utils.performance import PerformanceResult


@dataclass
class DriftTopology:
    """Classification of drift type and characteristics."""
    drift_type: str  # e.g., "Data Drift Only", "Concept Drift", "Combined", "No Drift"
    data_drift_severity: str  # "None", "Mild", "Strong"
    model_drift_severity: str  # "None", "Mild", "Strong"
    interpretation: str
    action_suggested: str


def analyze_drift_topology(
    drift_result: DriftResult = None,
    ref_perf: PerformanceResult = None,
    curr_perf: PerformanceResult = None,
    perf_threshold: float = 0.05,
) -> DriftTopology:
    """
    Analyze and classify the type and severity of drift detected.
    
    Args:
        drift_result: Result from detect_drift() (data drift metrics).
        ref_perf: Performance on reference data.
        curr_perf: Performance on current data.
        perf_threshold: Threshold for performance degradation (default 5%).
    
    Returns:
        DriftTopology with interpretation and suggested action.
    
    Logic:
        - Data drift strong: > 50% of features drifted, drift_pct ≥ 0.5
        - Performance drop: any metric degraded by ≥ threshold
        
        Profile 1: Data drift strong + perf stable → Model robust, ecosystem changed
        Profile 2: Data drift mild/none + perf drops → Likely concept drift (X->y relation changed)
        Profile 3: Data drift strong + perf drops → Serious drift, urgent retrain
        Profile 4: No drift, perf stable → No action needed
    """
    
    # Assess data drift severity
    if drift_result is None:
        data_drift_severity = "Unknown"
        overall_drift_detected = False
    else:
        overall_drift_detected = drift_result.overall_drift
        if drift_result.drift_pct >= 0.5:
            data_drift_severity = "Strong"
        elif drift_result.drift_pct >= 0.2:
            data_drift_severity = "Mild"
        else:
            data_drift_severity = "None"
    
    # Assess model/performance drift severity
    if ref_perf is None or curr_perf is None:
        model_drift_severity = "Unknown"
        perf_dropped = False
    else:
        perf_dropped = False
        max_drop = 0.0
        
        for metric_name, ref_val in ref_perf.metrics.items():
            curr_val = curr_perf.metrics.get(metric_name)
            if curr_val is None:
                continue
            
            # For error metrics (MSE, RMSE, MAE), increase = worse
            if metric_name in ("MSE", "RMSE", "MAE"):
                change = (curr_val - ref_val) / max(abs(ref_val), 1e-8)
            else:
                # For performance metrics (Accuracy, F1, R2), decrease = worse
                change = (ref_val - curr_val) / max(abs(ref_val), 1e-8)
            
            max_drop = max(max_drop, change)
        
        perf_dropped = max_drop > perf_threshold
        
        if max_drop >= 0.15:
            model_drift_severity = "Strong"
        elif max_drop >= perf_threshold:
            model_drift_severity = "Mild"
        else:
            model_drift_severity = "None"
    
    # Classify drift topology
    if data_drift_severity == "Unknown" or model_drift_severity == "Unknown":
        drift_type = "Unknown"
        interpretation = "Insufficient data to assess drift."
        action_suggested = "Collect more data or check inputs."
    
    elif data_drift_severity == "None" and model_drift_severity == "None":
        drift_type = "No Drift"
        interpretation = "System is stable. No significant drift detected in data or model performance."
        action_suggested = "Continue monitoring."
    
    elif data_drift_severity == "Strong" and model_drift_severity == "None":
        drift_type = "Data Drift Only (Model Robust)"
        interpretation = (
            "Data distribution has changed significantly, but the model's performance "
            "remains stable. This suggests the model is robust to distributional changes."
        )
        action_suggested = "Monitor closely; retrain optional but not urgent."
    
    elif data_drift_severity in ("Mild", "None") and model_drift_severity in ("Mild", "Strong"):
        drift_type = "Concept Drift (Primary)"
        interpretation = (
            "The relationship between features and target has likely changed (concept drift). "
            "Data distribution change is minimal, but model performance is degrading. "
            "This indicates the decision boundary or underlying pattern has shifted."
        )
        action_suggested = "Retrain recommended to adapt to new concept."
    
    elif data_drift_severity == "Strong" and model_drift_severity in ("Mild", "Strong"):
        drift_type = "Combined Drift (Data + Concept)"
        interpretation = (
            "Both data distribution and the relationship between features and target have changed. "
            "This is the most concerning scenario, indicating a fundamental shift in the problem domain."
        )
        action_suggested = "Urgent retrain + investigation of root cause."
    
    else:
        drift_type = "Mixed"
        interpretation = "Complex drift pattern detected."
        action_suggested = "Investigate further."
    
    return DriftTopology(
        drift_type=drift_type,
        data_drift_severity=data_drift_severity,
        model_drift_severity=model_drift_severity,
        interpretation=interpretation,
        action_suggested=action_suggested,
    )
