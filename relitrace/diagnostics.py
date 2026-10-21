"""Feature-level, model-independent comparison of two tabular datasets.

These are descriptive shift scores, not significance tests or model-safety checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def detect_data_shift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    distribution_threshold: float = 0.10,
    missingness_threshold: float = 0.05,
) -> pd.DataFrame:
    """Compare feature distributions and missingness using heuristic cutoffs.

    Both inputs must have matching named, uniquely-labelled columns.
    Numeric columns use the two-sample KS statistic on nonmissing values;
    categoricals use total variation distance including missing as a category.
    Missing rates are also checked independently. A flag is not model failure.
    """
    for name, frame in (("reference", reference), ("current", current)):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        if frame.empty:
            raise ValueError(f"{name} must not be empty")
        if frame.columns.has_duplicates:
            raise ValueError(f"{name} has duplicate column names")
        if not all(isinstance(c, str) and c.strip() for c in frame.columns):
            raise ValueError(f"{name} must have nonempty string column names")
    if not reference.columns.equals(current.columns):
        raise ValueError("Reference and current columns or their order differ")
    if not np.isfinite(distribution_threshold) or not 0 < distribution_threshold <= 1:
        raise ValueError("Distribution threshold must be greater than 0 and at most 1")
    if not np.isfinite(missingness_threshold) or not 0 < missingness_threshold <= 1:
        raise ValueError("Missingness threshold must be greater than 0 and at most 1")

    rows = []
    for feature in reference.columns:
        ref = reference[feature]
        cur = current[feature]
        numeric = pd.api.types.is_numeric_dtype(ref) and not pd.api.types.is_bool_dtype(ref)
        current_numeric = pd.api.types.is_numeric_dtype(cur) and not pd.api.types.is_bool_dtype(cur)
        if numeric != current_numeric:
            raise TypeError(f"Feature '{feature}' has incompatible numeric/categorical types")
        reference_missing = float(ref.isna().mean())
        current_missing = float(cur.isna().mean())
        missing_delta = current_missing - reference_missing

        if numeric:
            ref_values = ref.dropna()
            cur_values = cur.dropna()
            score = float(ks_2samp(ref_values, cur_values).statistic) if len(ref_values) and len(cur_values) else np.nan
            method = "KS statistic"
        else:
            ref_probs = ref.astype('string').fillna('<MISSING>').value_counts(normalize=True)
            cur_probs = cur.astype('string').fillna('<MISSING>').value_counts(normalize=True)
            categories = ref_probs.index.union(cur_probs.index)
            score = float(0.5 * (
                ref_probs.reindex(categories, fill_value=0) -
                cur_probs.reindex(categories, fill_value=0)
            ).abs().sum())
            method = "Total variation distance"

        distribution_flag = bool(pd.notna(score) and score > distribution_threshold)
        missing_flag = bool(abs(missing_delta) > missingness_threshold)
        rows.append({
            "feature": feature,
            "method": method,
            "distribution_score": score,
            "reference_missing_rate": reference_missing,
            "current_missing_rate": current_missing,
            "missing_change": missing_delta,
            "distribution_flag": distribution_flag,
            "missing_flag": missing_flag,
            "shift_detected": distribution_flag or missing_flag,
        })
    return pd.DataFrame(rows)
