import pytest
from relitrace.csv_input import align_feature_frames, read_feature_csv


def test_valid_csv():
    frame = read_feature_csv(b"age,job\n20,admin\n30,tech\n")
    assert frame.shape == (2, 2)


def test_reorders_columns():
    ref = read_feature_csv(b"age,job\n20,admin\n")
    cur = read_feature_csv(b"job,age\ntech,30\n")
    ref, cur = align_feature_frames(ref, cur)
    assert list(cur.columns) == list(ref.columns)


def test_column_mismatch():
    ref = read_feature_csv(b"age,job\n20,admin\n")
    cur = read_feature_csv(b"age,salary\n30,100\n")
    with pytest.raises(ValueError):
        align_feature_frames(ref, cur)


def test_reject_duplicate_headers():
    with pytest.raises(ValueError):
        read_feature_csv(b"age,age\n20,30\n")


def test_reject_unnamed_index():
    with pytest.raises(ValueError):
        read_feature_csv(b",age\n0,20\n")


def test_reject_empty():
    with pytest.raises(ValueError):
        read_feature_csv(b"")
