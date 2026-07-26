"""Output-boundary helpers: build validator-safe rows and write JSONL.

This module is intentionally the only place that knows about the submission
schema and its validator-safe placeholder values. Extraction/adjudication
code upstream should be free to pass "unknown"/missing internal values;
build_row() is responsible for filling in placeholders that satisfy the
challenge's scripts/validate_submission.py constraints.
"""
import json
from pathlib import Path

# Canonical field order for every output row. This exact order is what gets
# written to JSONL so output is stable/reviewable.
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

# Validator-safe placeholders used whenever a field's real value is unknown
# or missing. adjudication/confidence/case_id are supplied by the caller of
# build_row(), not defaulted here.
FALLBACK_ROW = {
    "applicant_name": "unknown",
    "species_code": "unknown",
    "home_world": "unknown",
    "visa_class": "unknown",
    "sponsor_id": "SPN-0000",
    "arrival_date": "1900-01-01",
    "declared_purpose": "unknown",
    "risk_flags": "none",
    "fee_status": "unknown",
}


def build_row(case_id: str, fields: dict, adjudication: str, confidence: float) -> dict:
    """Build one validator-safe output row.

    `fields` may omit any subset of the non-identifying keys (or contain
    falsy/empty values); missing keys are filled from FALLBACK_ROW so the
    resulting row always has all 12 required fields with schema-valid types.
    """
    fields = fields or {}
    row = {"case_id": case_id}
    for key, default in FALLBACK_ROW.items():
        value = fields.get(key)
        row[key] = value if value not in (None, "") else default
    row["adjudication"] = adjudication
    row["confidence"] = confidence
    # Re-order to the canonical field order.
    return {name: row[name] for name in FIELDNAMES}


def write_jsonl(rows, path) -> None:
    """Write rows as JSONL, one compact JSON object per line, sorted by
    case_id for deterministic output. Overwrites `path` if it exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: r["case_id"])
    with open(path, "w") as f:
        for row in sorted_rows:
            ordered = {name: row[name] for name in FIELDNAMES}
            f.write(json.dumps(ordered, sort_keys=False))
            f.write("\n")
