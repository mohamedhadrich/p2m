"""Drift detection: PSI, KS test, Chi-squared test."""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field

# ── Sensitivity presets ──────────────────────────────────────────
SENSITIVITY_PRESETS = {
    "Strict": {
        "psi_threshold": 0.10,
        "ks_pvalue": 0.01,
        "feature_drift_pct": 0.30,
    },
    "Balanced": {
        "psi_threshold": 0.20,
        "ks_pvalue": 0.05,
        "feature_drift_pct": 0.50,
    },
    "Loose": {
        "psi_threshold": 0.25,
        "ks_pvalue": 0.10,
        "feature_drift_pct": 0.70,
    },
}


# ── Per-column statistics ────────────────────────────────────────
def compute_psi(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Population Stability Index between two numeric arrays."""
    eps = 1e-4
    breakpoints = np.linspace(np.min(reference), np.max(reference), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    curr_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * bins)
    curr_pct = (curr_counts + eps) / (curr_counts.sum() + eps * bins)

    psi = float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))
    return psi


def ks_test(reference: np.ndarray, current: np.ndarray):
    """Kolmogorov-Smirnov two-sample test."""
    stat, pvalue = stats.ks_2samp(reference, current)
    return float(stat), float(pvalue)


def chi2_test(reference: pd.Series, current: pd.Series):
    """Chi-squared test for categorical columns."""
    all_cats = sorted(set(reference.unique()) | set(current.unique()))
    ref_counts = reference.value_counts()
    curr_counts = current.value_counts()

    ref_arr = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
    curr_arr = np.array([curr_counts.get(c, 0) for c in all_cats], dtype=float)

    # Add 1 to avoid zeros
    stat, pvalue = stats.chisquare(curr_arr + 1, f_exp=ref_arr + 1)
    return float(stat), float(pvalue)


# ── Result data-classes ──────────────────────────────────────────
@dataclass
class ColumnDrift:
    column: str
    drift_detected: bool
    method: str
    statistic: float
    pvalue: float = None
    psi: float = None


@dataclass
class DriftResult:
    overall_drift: bool
    n_drifted: int
    n_total: int
    drift_pct: float
    column_results: list = field(default_factory=list)
    sensitivity: str = "Balanced"


# ── Main entry point ─────────────────────────────────────────────
def detect_drift(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    columns: list,
    sensitivity: str = "Balanced",
    target: str = None,
) -> DriftResult:
    """Run drift detection on all *columns* (excluding target)."""
    preset = SENSITIVITY_PRESETS[sensitivity]
    feature_cols = [c for c in columns if c != target]

    column_results: list[ColumnDrift] = []
    n_drifted = 0

    for col in feature_cols:
        ref_s = ref_df[col].dropna()
        curr_s = curr_df[col].dropna()
        if len(ref_s) == 0 or len(curr_s) == 0:
            continue

        if pd.api.types.is_numeric_dtype(ref_df[col]):
            psi = compute_psi(ref_s.values, curr_s.values)
            ks_stat, ks_pval = ks_test(ref_s.values, curr_s.values)
            drifted = psi > preset["psi_threshold"] or ks_pval < preset["ks_pvalue"]
            column_results.append(
                ColumnDrift(
                    column=col,
                    drift_detected=drifted,
                    method="PSI + KS",
                    statistic=ks_stat,
                    pvalue=ks_pval,
                    psi=psi,
                )
            )
        else:
            chi_stat, chi_pval = chi2_test(ref_s, curr_s)
            drifted = chi_pval < preset["ks_pvalue"]
            column_results.append(
                ColumnDrift(
                    column=col,
                    drift_detected=drifted,
                    method="Chi²",
                    statistic=chi_stat,
                    pvalue=chi_pval,
                )
            )

        if drifted:
            n_drifted += 1

    n_total = len(column_results)
    drift_pct = n_drifted / n_total if n_total > 0 else 0.0
    overall_drift = drift_pct >= preset["feature_drift_pct"]

    return DriftResult(
        overall_drift=overall_drift,
        n_drifted=n_drifted,
        n_total=n_total,
        drift_pct=drift_pct,
        column_results=column_results,
        sensitivity=sensitivity,
    )
