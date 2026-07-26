"""Tests for mib.output: build_row / write_jsonl.

The row-validation helper below is a local copy of the constraints enforced by
the challenge repo's scripts/validate_submission.py (validate_row). It is
duplicated here (not imported) because the challenge repo is a read-only
reference and not a dependency of this solution.
"""
import json
import math
import re
from datetime import date
from pathlib import Path

from mib.output import build_row, write_jsonl, FALLBACK_ROW

FIELDNAMES = [
    "case_id",
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
    "adjudication",
    "confidence",
]
STRING_FIELDS = [f for f in FIELDNAMES if f != "confidence"]
FEE_VALUES = {"paid", "waived", "unpaid", "unknown"}
ADJUDICATION_VALUES = {"APPROVED", "DENIED", "NEEDS_REVIEW"}
CASE_ID_PATTERN = re.compile(r"^MIB-[0-9]{6}$")
SPONSOR_ID_PATTERN = re.compile(r"^SPN-[0-9]{4}$")


def validate_row(row):
    """Local copy of the challenge validator's per-row checks. Returns a list
    of error strings; empty list means the row is valid."""
    errors = []

    missing = [f for f in FIELDNAMES if f not in row]
    extra = sorted(set(row) - set(FIELDNAMES))
    if missing:
        errors.append(f"missing fields {missing}")
    if extra:
        errors.append(f"unexpected fields {extra}")
    if missing:
        return errors

    for field in STRING_FIELDS:
        if not isinstance(row[field], str):
            errors.append(f"{field} must be a string")

    case_id = row["case_id"].strip() if isinstance(row["case_id"], str) else ""
    if not CASE_ID_PATTERN.fullmatch(case_id):
        errors.append(f"invalid case_id {row['case_id']!r}")

    sponsor_id = row["sponsor_id"].strip() if isinstance(row["sponsor_id"], str) else ""
    if not SPONSOR_ID_PATTERN.fullmatch(sponsor_id):
        errors.append(f"invalid sponsor_id {row['sponsor_id']!r}")

    arrival_date = row["arrival_date"].strip() if isinstance(row["arrival_date"], str) else ""
    try:
        parsed = date.fromisoformat(arrival_date)
    except ValueError:
        errors.append(f"invalid arrival_date {row['arrival_date']!r}")
    else:
        if parsed.isoformat() != arrival_date:
            errors.append(f"invalid arrival_date {row['arrival_date']!r}")

    fee_status = row["fee_status"].strip() if isinstance(row["fee_status"], str) else ""
    if isinstance(row["fee_status"], str) and fee_status not in FEE_VALUES:
        errors.append(f"invalid fee_status {row['fee_status']!r}")

    adjudication = row["adjudication"].strip() if isinstance(row["adjudication"], str) else ""
    if isinstance(row["adjudication"], str) and adjudication not in ADJUDICATION_VALUES:
        errors.append(f"invalid adjudication {row['adjudication']!r}")

    if isinstance(row["confidence"], bool) or not isinstance(row["confidence"], (int, float)):
        errors.append("confidence must be a JSON number")
        return errors
    confidence = float(row["confidence"])
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        errors.append("confidence must be between 0 and 1")

    return errors


def test_build_row_all_unknown_passes_validator():
    row = build_row("MIB-000001", {}, "NEEDS_REVIEW", 0.3)
    errors = validate_row(row)
    assert errors == [], errors


def test_build_row_has_all_twelve_fields():
    row = build_row("MIB-000001", {}, "NEEDS_REVIEW", 0.3)
    assert set(row.keys()) == set(FIELDNAMES)


def test_build_row_fallback_placeholders():
    row = build_row("MIB-000001", {}, "NEEDS_REVIEW", 0.3)
    assert row["applicant_name"] == "unknown"
    assert row["species_code"] == "unknown"
    assert row["home_world"] == "unknown"
    assert row["visa_class"] == "unknown"
    assert row["declared_purpose"] == "unknown"
    assert row["sponsor_id"] == "SPN-0000"
    assert row["arrival_date"] == "1900-01-01"
    assert row["risk_flags"] == "none"
    assert row["fee_status"] == "unknown"
    assert row["adjudication"] == "NEEDS_REVIEW"
    assert row["confidence"] == 0.3
    assert row["case_id"] == "MIB-000001"


def test_build_row_uses_provided_fields_over_fallback():
    fields = {
        "applicant_name": "Zog Blorptun",
        "species_code": "GRX",
        "home_world": "Kepler-9c",
        "visa_class": "B2",
        "sponsor_id": "SPN-1234",
        "arrival_date": "2025-01-15",
        "declared_purpose": "trade",
        "risk_flags": "overstay",
        "fee_status": "paid",
    }
    row = build_row("MIB-000002", fields, "APPROVED", 0.9)
    errors = validate_row(row)
    assert errors == [], errors
    for key, value in fields.items():
        assert row[key] == value
    assert row["adjudication"] == "APPROVED"
    assert row["confidence"] == 0.9


def test_build_row_rejects_invalid_case_id_shape_via_validator():
    # Sanity check: the validator's own pattern should reject malformed ids.
    row = build_row("MIB-1", {}, "NEEDS_REVIEW", 0.3)
    errors = validate_row(row)
    assert any("invalid case_id" in e for e in errors)


def test_fallback_row_constants():
    assert FALLBACK_ROW["applicant_name"] == "unknown"
    assert FALLBACK_ROW["species_code"] == "unknown"
    assert FALLBACK_ROW["home_world"] == "unknown"
    assert FALLBACK_ROW["visa_class"] == "unknown"
    assert FALLBACK_ROW["declared_purpose"] == "unknown"
    assert FALLBACK_ROW["sponsor_id"] == "SPN-0000"
    assert FALLBACK_ROW["arrival_date"] == "1900-01-01"
    assert FALLBACK_ROW["risk_flags"] == "none"
    assert FALLBACK_ROW["fee_status"] == "unknown"


def test_write_jsonl_writes_one_json_object_per_line(tmp_path):
    rows = [
        build_row("MIB-000002", {}, "NEEDS_REVIEW", 0.3),
        build_row("MIB-000001", {}, "NEEDS_REVIEW", 0.3),
    ]
    out_path = tmp_path / "predictions.jsonl"
    write_jsonl(rows, out_path)

    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert all(isinstance(p, dict) for p in parsed)


def test_write_jsonl_sorted_by_case_id(tmp_path):
    rows = [
        build_row("MIB-000003", {}, "NEEDS_REVIEW", 0.3),
        build_row("MIB-000001", {}, "NEEDS_REVIEW", 0.3),
        build_row("MIB-000002", {}, "NEEDS_REVIEW", 0.3),
    ]
    out_path = tmp_path / "predictions.jsonl"
    write_jsonl(rows, out_path)

    parsed = [json.loads(line) for line in out_path.read_text().splitlines()]
    ids = [p["case_id"] for p in parsed]
    assert ids == sorted(ids)


def test_write_jsonl_deterministic_key_order(tmp_path):
    row = build_row("MIB-000001", {}, "NEEDS_REVIEW", 0.3)
    out_path = tmp_path / "predictions.jsonl"
    write_jsonl([row], out_path)
    line = out_path.read_text().splitlines()[0]
    parsed_keys = list(json.loads(line).keys())
    assert parsed_keys == FIELDNAMES
