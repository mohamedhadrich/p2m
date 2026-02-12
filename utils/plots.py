"""Plotly charts for the Streamlit dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_distribution_comparison(
    ref_series: pd.Series, curr_series: pd.Series, col_name: str
) -> go.Figure:
    """Overlapping histograms: reference vs current."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=ref_series, name="Reference", opacity=0.6, marker_color="#636EFA")
    )
    fig.add_trace(
        go.Histogram(x=curr_series, name="Current", opacity=0.6, marker_color="#EF553B")
    )
    fig.update_layout(
        title=f"Distribution — {col_name}",
        barmode="overlay",
        xaxis_title=col_name,
        yaxis_title="Count",
        template="plotly_white",
        height=350,
    )
    return fig


def plot_drift_summary(drift_result) -> go.Figure | None:
    """Horizontal bar chart of PSI per feature (numeric features only)."""
    data = []
    for cr in drift_result.column_results:
        data.append(
            {
                "Column": cr.column,
                "PSI": cr.psi if cr.psi is not None else 0,
                "Drifted": "Yes" if cr.drift_detected else "No",
            }
        )
    if not data:
        return None
    df = pd.DataFrame(data).sort_values("PSI", ascending=True)
    fig = px.bar(
        df,
        y="Column",
        x="PSI",
        color="Drifted",
        orientation="h",
        color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        title="Drift Summary — PSI per feature",
        template="plotly_white",
    )
    fig.update_layout(height=max(300, len(data) * 28 + 120))
    return fig


def plot_missing_values(missing_df: pd.DataFrame) -> go.Figure | None:
    """Bar chart of missing-value percentages."""
    if len(missing_df) == 0:
        return None
    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing %",
        title="Missing Values by Column",
        template="plotly_white",
        color="Missing %",
        color_continuous_scale="Reds",
    )
    return fig
