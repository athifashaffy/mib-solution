"""End-to-end hardening tests for mib.main."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_corrupted_pdf_yields_fallback_row(tmp_path):
    bad = tmp_path / "MIB-999901.pdf"
    bad.write_bytes(b"%PDF-1.4 truncated garbage")
    out = tmp_path / "predictions.jsonl"
    proc = subprocess.run(
        [sys.executable, "-m", "mib.main", str(tmp_path), str(out)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["case_id"] == "MIB-999901"
    assert row["adjudication"] == "NEEDS_REVIEW"
    assert 0.0 <= row["confidence"] <= 1.0


def test_non_pdf_and_bad_stem_files_are_ignored(tmp_path):
    (tmp_path / "notes.txt").write_text("not a pdf")
    (tmp_path / "random-name.pdf").write_bytes(b"%PDF-1.4")
    out = tmp_path / "predictions.jsonl"
    proc = subprocess.run(
        [sys.executable, "-m", "mib.main", str(tmp_path), str(out)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.read_text() == ""
