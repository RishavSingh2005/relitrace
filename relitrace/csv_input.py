"""Conservative CSV ingestion for the public-facing diagnostic prototype."""
from __future__ import annotations
from io import BytesIO
import pandas as pd

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 200


def read_feature_csv(content: bytes, filename: str = "CSV") -> pd.DataFrame:
    """Read tabular features only; reject empty, duplicate, or oversized inputs."""
    if not content:
        raise ValueError(f"{filename} is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"{filename} exceeds the 20 MB limit")
    try:
        frame = pd.read_csv(BytesIO(content), low_memory=False)
    except (UnicodeError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Could not read {filename}: {exc}") from exc
    if frame.empty:
        raise ValueError(f"{filename} has no rows or columns")
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise ValueError(f"{filename} exceeds {MAX_ROWS:,} rows or {MAX_COLUMNS} columns")
    if any(str(col).startswith("Unnamed:") for col in frame.columns):
        raise ValueError(f"{filename} contains unnamed columns. Export with index=False")
    # pandas mangles duplicate headers by adding '.1'; detect from CSV header instead.
    try:
        import csv
        first_line = content.splitlines()[0].decode("utf-8-sig")
        header = next(csv.reader([first_line]))
    except (UnicodeError, StopIteration, csv.Error) as exc:
        raise ValueError(f"Could not read header in {filename}: {exc}") from exc
    if len(header) != len(set(header)):
        raise ValueError(f"{filename} has duplicate CSV header names")
    if any(not h.strip() for h in header):
        raise ValueError(f"{filename} has blank CSV header names")
    return frame


def align_feature_frames(reference: pd.DataFrame, current: pd.DataFrame):
    """Ensure names match and align current features to reference order."""
    missing = sorted(set(reference.columns) - set(current.columns))
    extra = sorted(set(current.columns) - set(reference.columns))
    if missing or extra:
        raise ValueError(
            f"Feature schema mismatch. Missing in current: {missing[:10]}; "
            f"unexpected in current: {extra[:10]}"
        )
    return reference, current.loc[:, reference.columns].copy()
