import numpy as np
import pandas as pd
import pytest
from relitrace.calibration import calibrate_thresholds
from relitrace.diagnostics import detect_data_shift


def example(n=300):
    rng = np.random.default_rng(77)
    return pd.DataFrame({
        "age": rng.normal(40, 8, n),
        "balance": rng.exponential(100, n),
        "job": rng.choice(["tech", "admin", "other"], n),
    })


def test_auto_thresholds_are_positive_and_at_most_one():
    result = calibrate_thresholds(example(), 75, calibration_runs=30, validation_runs=10)
    assert 0 < result.distribution_threshold <= 1
    assert 0 < result.missingness_threshold <= 1
    assert 0 <= result.observed_control_warning_fraction <= 1
    assert len(result.validation_scores) == 10


def test_reproducible_seed():
    ref = example()
    a = calibrate_thresholds(ref, 75, calibration_runs=30, validation_runs=10, seed=20)
    b = calibrate_thresholds(ref, 75, calibration_runs=30, validation_runs=10, seed=20)
    assert a.distribution_threshold == b.distribution_threshold
    assert a.missingness_threshold == b.missingness_threshold
    pd.testing.assert_frame_equal(a.validation_scores, b.validation_scores)


def test_detect_intentional_numeric_shift_after_calibration():
    ref = example()
    result = calibrate_thresholds(ref, 75, calibration_runs=30, validation_runs=10)
    cur = ref.iloc[:75].copy()
    cur["age"] += 80
    report = detect_data_shift(ref, cur, result.distribution_threshold,
                               result.missingness_threshold)
    assert bool(report.set_index("feature").loc["age", "shift_detected"])


def test_detect_intentional_missingness_after_calibration():
    ref = example()
    result = calibrate_thresholds(ref, 75, calibration_runs=30, validation_runs=10)
    cur = ref.iloc[:75].copy()
    cur.loc[cur.index[:35], "balance"] = np.nan
    report = detect_data_shift(ref, cur, result.distribution_threshold,
                               result.missingness_threshold)
    assert bool(report.set_index("feature").loc["balance", "missing_flag"])


def test_too_few_reference_rows():
    with pytest.raises(ValueError, match="80 reference"):
        calibrate_thresholds(example(70), 20)


def test_invalid_run_count_rejected():
    with pytest.raises(ValueError, match="Calibration runs"):
        calibrate_thresholds(example(), 60, calibration_runs=5)


def test_reference_unchanged():
    ref = example()
    before = ref.copy(deep=True)
    calibrate_thresholds(ref, 75, calibration_runs=30, validation_runs=10)
    pd.testing.assert_frame_equal(ref, before)


def test_small_current_sample_uses_all_rows():
    result = calibrate_thresholds(example(), 24, calibration_runs=30, validation_runs=10)
    assert result.current_sample_rows == 24
    assert result.reference_sample_rows == 276
