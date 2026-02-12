"""Schema comparison between reference and current datasets."""

import pandas as pd
from dataclasses import dataclass


@dataclass
class SchemaComparison:
    common_columns: list
    ref_only_columns: list
    curr_only_columns: list
    type_mismatches: dict        # col -> (ref_type, curr_type)
    is_compatible: bool


def compare_schemas(
    ref_df: pd.DataFrame, curr_df: pd.DataFrame
) -> SchemaComparison:
    """Compare column names and types between two DataFrames."""
    ref_cols = set(ref_df.columns)
    curr_cols = set(curr_df.columns)

    common = sorted(ref_cols & curr_cols)
    ref_only = sorted(ref_cols - curr_cols)
    curr_only = sorted(curr_cols - ref_cols)

    type_mismatches = {}
    for col in common:
        if ref_df[col].dtype != curr_df[col].dtype:
            type_mismatches[col] = (str(ref_df[col].dtype), str(curr_df[col].dtype))

    is_compatible = (
        len(common) > 0
        and len(ref_only) == 0
        and len(curr_only) == 0
        and len(type_mismatches) == 0
    )

    return SchemaComparison(
        common_columns=common,
        ref_only_columns=ref_only,
        curr_only_columns=curr_only,
        type_mismatches=type_mismatches,
        is_compatible=is_compatible,
    )
