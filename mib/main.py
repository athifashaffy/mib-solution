"""CLI entry point: python -m mib.main <input_dir> <output_path>

Task 1 scope: an end-to-end skeleton. Every PDF found in <input_dir> gets a
single validator-safe NEEDS_REVIEW row with all-fallback fields. Extraction
and adjudication logic land in later tasks; this file's job is to prove the
container contract end to end (read PDFs, never crash, always emit exactly
one row per input case, deterministic sorted output).
"""
import re
import sys
from pathlib import Path

from mib.output import build_row, write_jsonl

CASE_ID_PATTERN = re.compile(r"^MIB-[0-9]{6}$")


def process_dir(input_dir, output_path) -> None:
    """Serially process every *.pdf in input_dir and write one row per case
    to output_path as JSONL, sorted by case_id."""
    input_dir = Path(input_dir)
    rows = []
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        case_id = pdf_path.stem
        if not CASE_ID_PATTERN.fullmatch(case_id):
            # Only case IDs matching the expected shape are valid submission
            # rows; skip anything else rather than emit a row the validator
            # will reject.
            continue
        row = build_row(case_id, {}, "NEEDS_REVIEW", 0.3)
        rows.append(row)
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
