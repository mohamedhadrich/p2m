"""Data quality checks: types, missing values, outliers."""

import pandas as pd
import numpy as np


def check_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return type information for every column."""
    rows = []
    for col in df.columns:
        rows.append(
            {
                "Column": col,
                "Type": str(df[col].dtype),
                "Numeric": pd.api.types.is_numeric_dtype(df[col]),
                "Unique Values": df[col].nunique(),
            }
        )
    return pd.DataFrame(rows)


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns that have missing values (count + %)."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    result = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": pct.values,
        }
    )
    return result[result["Missing Count"] > 0].reset_index(drop=True)


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers with IQR method on numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rows = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_outliers > 0:
            rows.append(
                {
                    "Column": col,
                    "Outliers": n_outliers,
                    "Outlier %": round(n_outliers / len(df) * 100, 2),
                    "Lower Bound": round(lower, 2),
                    "Upper Bound": round(upper, 2),
                }
            )
    return pd.DataFrame(rows)
