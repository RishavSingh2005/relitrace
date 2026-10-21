"""Reference-only exploratory calibration for multi-feature drift warnings.

Disjoint sample partitions *within each trial*, but observations are reused
across trials. The empirical warning fraction is descriptive, not a guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .diagnostics import detect_data_shift


@dataclass(frozen=True)
class CalibrationResult:
    distribution_threshold: float
    missingness_threshold: float
    null_scores: pd.DataFrame
    validation_scores: pd.DataFrame
    observed_control_warning_fraction: float
    reference_sample_rows: int
    current_sample_rows: int
    calibration_runs: int
    validation_runs: int
    alpha: float
    seed: int


def _null_scores(reference: pd.DataFrame, current_n: int, runs: int,
                 rng: np.random.Generator) -> pd.DataFrame:
    n = len(reference)
    records = []
    for run in range(runs):
        order = rng.permutation(n)
        current_sample = reference.iloc[order[:current_n]]
        reference_sample = reference.iloc[order[current_n:]]
        diagnostics = detect_data_shift(reference_sample, current_sample)
        records.append({
            "run": run + 1,
            "max_distribution_score": float(
                diagnostics["distribution_score"].fillna(0).max()
            ),
            "max_missingness_change": float(
                diagnostics["missing_change"].abs().max()
            ),
        })
    return pd.DataFrame(records)


def calibrate_thresholds(reference: pd.DataFrame, current_rows: int,
                         *, calibration_runs: int = 100,
                         validation_runs: int = 40,
                         alpha: float = 0.05,
                         seed: int = 2026) -> CalibrationResult:
    """Estimate two family-wise cutoffs using only *reference* data.

    A 1-alpha/2 empirical quantile is used for each warning family
    (distribution and missingness). This is a Bonferroni-style heuristic,
    not a formal error-control guarantee. Validate on separate real data.
    """
    if not isinstance(reference, pd.DataFrame) or reference.empty:
        raise ValueError("Reference must be a nonempty pandas DataFrame")
    if not isinstance(current_rows, (int, np.integer)) or current_rows < 1:
        raise ValueError("Current dataset must have at least one row")
    if len(reference) < 80:
        raise ValueError(
            "Auto calibration requires at least 80 reference rows; "
            "use manual thresholds for smaller datasets."
        )
    if not isinstance(calibration_runs, int) or not 20 <= calibration_runs <= 500:
        raise ValueError("Calibration runs must be between 20 and 500")
    if not isinstance(validation_runs, int) or not 10 <= validation_runs <= 200:
        raise ValueError("Validation runs must be between 10 and 200")
    if not isinstance(alpha, (int, float)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")

    # Match actual current sample size where feasible. Preserve at least 2/3
    # of the reference rows in each synthetic reference partition.
    current_n = min(int(current_rows), len(reference) // 3)
    rng = np.random.default_rng(seed)
    calibration = _null_scores(reference, current_n, calibration_runs, rng)
    quantile = 1.0 - alpha / 2.0
    # Tiny positive floor is needed because the detector enforces thresholds >0.
    dist_cutoff = float(max(1e-6, calibration["max_distribution_score"].quantile(quantile)))
    miss_cutoff = float(max(1e-6, calibration["max_missingness_change"].quantile(quantile)))
    dist_cutoff = min(dist_cutoff, 1.0)
    miss_cutoff = min(miss_cutoff, 1.0)

    # No reuse of validation *partitions* from calibration, but same dataset.
    validation = _null_scores(reference, current_n, validation_runs, rng)
    validation["distribution_warning"] = validation["max_distribution_score"] > dist_cutoff
    validation["missingness_warning"] = validation["max_missingness_change"] > miss_cutoff
    validation["any_warning"] = validation["distribution_warning"] | validation["missingness_warning"]
    return CalibrationResult(
        distribution_threshold=dist_cutoff,
        missingness_threshold=miss_cutoff,
        null_scores=calibration,
        validation_scores=validation,
        observed_control_warning_fraction=float(validation["any_warning"].mean()),
        reference_sample_rows=len(reference) - current_n,
        current_sample_rows=current_n,
        calibration_runs=calibration_runs,
        validation_runs=validation_runs,
        alpha=float(alpha),
        seed=seed,
    )
