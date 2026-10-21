from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_entrypoints_present():
    assert (ROOT / "app.py").is_file()
    assert (ROOT / "pages/1_Run_Diagnostics.py").is_file()
    assert (ROOT / "requirements.txt").is_file()


def test_streamlit_max_upload_size_matches_csv_limit():
    import tomllib
    with (ROOT / ".streamlit/config.toml").open("rb") as f:
        cfg = tomllib.load(f)
    from relitrace.csv_input import MAX_UPLOAD_BYTES
    assert cfg["server"]["maxUploadSize"] * 1024 * 1024 == MAX_UPLOAD_BYTES


def test_ci_and_gitignore_present():
    assert (ROOT / ".github/workflows/tests.yml").is_file()
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "*.csv" in ignored
    assert ".env" in ignored
