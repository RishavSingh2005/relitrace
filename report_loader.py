"""Read an allowlisted collection of ReliTrace aggregate experiment reports.

Never extracts ZIP contents or evaluates Python/pickle files. This parser loads
only expected CSV/JSON filenames into memory with conservative size limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
import json
import zipfile

import pandas as pd

REPORT_FILES = {
    "reliability_summary.csv": "reliability",
    "feature_diagnostics.csv": "features",
    "calibration_controls.csv": "calibration",
    "validation_controls.csv": "controls",
    "uncertainty_summary.csv": "uncertainty",
    "subgroup_diagnostics.csv": "subgroups",
    "experiment_metadata.json": "metadata",
}
OPTIONAL_FILES = {
    "baseline_metrics.csv": "baseline",
    "validation_summary.csv": "validation_summary",
    "shift_detection_summary.csv": "shift_summary",
}
MAX_ZIP_BYTES = 16 * 1024 * 1024
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED = 24 * 1024 * 1024
MAX_ZIP_MEMBERS = 200


@dataclass
class ReportBundle:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def has_data(self) -> bool:
        return bool(self.tables)


def _expected_name(raw_name: str) -> str | None:
    path = PurePosixPath(raw_name.replace("\\", "/"))
    parts = path.parts
    if ".." in parts or path.is_absolute() or not parts:
        return None
    filename = parts[-1]
    if filename not in REPORT_FILES and filename not in OPTIONAL_FILES:
        return None
    if filename in REPORT_FILES and filename != "experiment_metadata.json":
        if len(parts) < 2 or parts[-2] != "reports":
            return None
    # Optional reports are stored immediately under relitrace_outputs.
    if filename in OPTIONAL_FILES:
        if len(parts) < 2 or parts[-2] != "relitrace_outputs":
            return None
    return filename


def _store(name: str, payload: bytes, bundle: ReportBundle) -> None:
    key = REPORT_FILES.get(name, OPTIONAL_FILES.get(name))
    if key == "metadata":
        try:
            obj = json.loads(payload.decode("utf-8-sig"))
            if not isinstance(obj, dict):
                raise ValueError("Metadata must be a JSON object")
            bundle.metadata = obj
        except (UnicodeError, ValueError) as exc:
            bundle.warnings.append(f"Could not read {name}: {exc}")
        return
    try:
        frame = pd.read_csv(BytesIO(payload))
        if frame.empty:
            bundle.warnings.append(f"{name} contains no records")
        bundle.tables[key] = frame
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        bundle.warnings.append(f"Could not read {name}: {exc}")


def load_zip(payload: bytes) -> ReportBundle:
    bundle = ReportBundle()
    if len(payload) > MAX_ZIP_BYTES:
        bundle.warnings.append("Archive too large (limit 16 MB)")
        return bundle
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) > MAX_ZIP_MEMBERS:
                bundle.warnings.append("Archive contains too many entries")
                return bundle
            if sum(m.file_size for m in members) > MAX_TOTAL_UNCOMPRESSED:
                bundle.warnings.append("Archive uncompressed content exceeds limit")
                return bundle
            seen = set()
            for info in members:
                name = _expected_name(info.filename)
                if not name or info.is_dir():
                    continue
                if name in seen:
                    bundle.warnings.append(f"Duplicate report ignored: {name}")
                    continue
                seen.add(name)
                if info.file_size > MAX_FILE_BYTES:
                    bundle.warnings.append(f"Report exceeds 4 MB: {name}")
                    continue
                try:
                    _store(name, archive.read(info), bundle)
                except (RuntimeError, OSError, zipfile.BadZipFile) as exc:
                    bundle.warnings.append(f"Could not open {name}: {exc}")
    except (zipfile.BadZipFile, OSError) as exc:
        bundle.warnings.append(f"Invalid ZIP archive: {exc}")
    return bundle


def load_directory(project_root: Path) -> ReportBundle:
    bundle = ReportBundle()
    root = project_root / "relitrace_outputs"
    paths = {
        **{filename: root / "reports" / filename for filename in REPORT_FILES},
        **{filename: root / filename for filename in OPTIONAL_FILES},
    }
    for filename, path in paths.items():
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            bundle.warnings.append(f"Report exceeds 4 MB: {filename}")
            continue
        _store(filename, path.read_bytes(), bundle)
    return bundle
