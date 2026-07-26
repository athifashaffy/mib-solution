"""CLI entry point: python -m mib.main <input_dir> <output_path>

Full pipeline per PDF: extract (visible/hidden/OCR) -> resolve (precedence
merge, snapping) -> rules (cascade -> rule path) -> decide (expected-points
adjudication + calibrated confidence) -> validator-safe row.

Robustness contract: every PDF whose filename stem matches MIB-\\d{6} yields
exactly one row. Any per-PDF exception degrades to a NEEDS_REVIEW fallback
row — a missing case costs strictly more than a hedged one. Output is
deterministic and sorted by case_id.
"""
import re
import sys
from pathlib import Path

from mib.decide import decide, load_stats
from mib.extract import extract_document
from mib.output import build_row, write_jsonl
from mib.resolve import resolve_case
from mib.rules import infer_flags, path_of

CASE_ID_PATTERN = re.compile(r"^MIB-[0-9]{6}$")

FALLBACK_CONFIDENCE = 0.3


def process_pdf(pdf_path: Path, stats) -> dict:
    """Process one PDF into an output row. The case_id is always the filename
    stem (the packet's active case id — also what the scorer expects), never
    an extracted value, so a decoy id inside the document can't create an
    unexpected-case error."""
    case_id = pdf_path.stem
    try:
        doc = extract_document(str(pdf_path))
        record = resolve_case(doc)
        flags = infer_flags(record)
        adjudication, confidence = decide(path_of(record), stats)
        fields = dict(record.fields)
        fields["risk_flags"] = "|".join(sorted(flags)) if flags else "none"
        return build_row(case_id, fields, adjudication, confidence)
    except Exception:  # noqa: BLE001 - never omit a case
        return build_row(case_id, {}, "NEEDS_REVIEW", FALLBACK_CONFIDENCE)


def process_dir(input_dir, output_path) -> None:
    input_dir = Path(input_dir)
    stats = load_stats()
    rows = []
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        if not CASE_ID_PATTERN.fullmatch(pdf_path.stem):
            # A non-conforming id would make the whole submission invalid.
            continue
        rows.append(process_pdf(pdf_path, stats))
    write_jsonl(rows, output_path)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m mib.main <input_dir> <output_path>", file=sys.stderr)
        return 2
    input_dir, output_path = argv
    process_dir(input_dir, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
