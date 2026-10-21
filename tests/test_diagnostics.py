import numpy as np
import pandas as pd
import pytest
from relitrace.diagnostics import detect_data_shift


def toy():
    return pd.DataFrame({
        "age": [20, 22, 24, 26, 28, 30],
        "balance": [100., 200., 300., 400., 500., 600.],
        "job": ["admin", "tech", "admin", "tech", "admin", "tech"],
    })


def test_identical_has_no_warnings():
    df = toy()
    assert not detect_data_shift(df, df.copy())["shift_detected"].any()


def test_age_shift_detected():
    ref, cur = toy(), toy()
    cur["age"] += 50
    assert bool(detect_data_shift(ref, cur).set_index("feature").loc["age", "distribution_flag"])


def test_missingness_detected():
    ref, cur = toy(), toy()
    cur.loc[:2, "balance"] = np.nan
    assert bool(detect_data_shift(ref, cur).set_index("feature").loc["balance", "missing_flag"])


def test_category_change_detected():
    ref, cur = toy(), toy()
    cur["job"] = "unknown"
    assert bool(detect_data_shift(ref, cur).set_index("feature").loc["job", "distribution_flag"])


def test_no_mutation():
    ref, cur = toy(), toy()
    previous = ref.copy(deep=True)
    detect_data_shift(ref, cur)
    pd.testing.assert_frame_equal(ref, previous)


def test_schema_mismatch_rejected():
    with pytest.raises(ValueError):
        detect_data_shift(toy(), toy().drop(columns=["job"]))


def test_type_mismatch_rejected():
    cur = toy()
    cur["age"] = "unavailable"
    with pytest.raises(TypeError):
        detect_data_shift(toy(), cur)


def test_thresholds_reject_zero():
    with pytest.raises(ValueError):
        detect_data_shift(toy(), toy(), distribution_threshold=0)
