from io import BytesIO
from pathlib import Path
import zipfile
import pandas as pd
from report_loader import _expected_name, load_zip, load_directory


def bundle_zip(files):
    out = BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        for name, data in files.items():
            z.writestr(name, data)
    return out.getvalue()


def test_valid_reports_zip():
    data = bundle_zip({
        'relitrace_outputs/reports/reliability_summary.csv': 'experiment,f1_drop\noriginal,0.0\nage_shift,0.01\n',
        'relitrace_outputs/reports/experiment_metadata.json': '{"project":"ReliTrace"}',
    })
    bundle = load_zip(data)
    assert bundle.metadata['project'] == 'ReliTrace'
    assert bundle.tables['reliability'].shape == (2, 2)


def test_unknown_and_traversal_ignored():
    data = bundle_zip({'../../etc/passwd': 'password', 'fake.csv':'x,y\n1,2', 'reports/reliability_summary.csv': 'experiment\noriginal\n'})
    bundle = load_zip(data)
    assert list(bundle.tables) == ['reliability']


def test_bad_zip_is_warning():
    bundle = load_zip(b'not-a-zip')
    assert not bundle.has_data()
    assert bundle.warnings


def test_oversize_rejected():
    bundle = load_zip(b'x' * (16 * 1024 * 1024 + 1))
    assert not bundle.has_data()
    assert 'too large' in bundle.warnings[0]


def test_local_files(tmp_path):
    report_dir = tmp_path / 'relitrace_outputs' / 'reports'
    report_dir.mkdir(parents=True)
    (report_dir / 'uncertainty_summary.csv').write_text('experiment,mean_drop\nage_shift,0.01\n', encoding='utf-8')
    bundle = load_directory(tmp_path)
    assert float(bundle.tables['uncertainty']['mean_drop'].iloc[0]) == .01
