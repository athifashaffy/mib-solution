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
from multiprocessing import Pool, cpu_count
from pathlib import Path

from mib.decide import decide, load_stats
from mib.extract import extract_document
from mib.output import build_row, write_jsonl
from mib.resolve import resolve_case
from mib.rules import infer_flags, path_of

CASE_ID_PATTERN = re.compile(r"^MIB-[0-9]{6}$")

FALLBACK_CONFIDENCE = 0.3


def _process_pdf_str(pdf_path_str: str) -> dict:
    """Pool worker wrapper: loads stats lazily per process (cached by
    load_stats) and never raises."""
    return process_pdf(Path(pdf_path_str), load_stats())


def process_pdf(pdf_path: Path, stats) -> dict:
    """Process one PDF into an output row. The case_id is always the filename
    stem (the packet's active case id — also what the scorer expects), never
    an extracted value, so a decoy id inside the document can't create an
    unexpected-case error."""
    case_id = pdf_path.stem
    try:
        doc = extract_document(str(pdf_path))
        record = resolve_case(doc, active_case_id=case_id, pdf_path=str(pdf_path))
        flags = infer_flags(record)
        adjudication, confidence = decide(path_of(record), stats)
        fields = dict(record.fields)
        fields["risk_flags"] = "|".join(sorted(flags)) if flags else "none"
        return build_row(case_id, fields, adjudication, confidence)
    except Exception:  # noqa: BLE001 - never omit a case
        return build_row(case_id, {}, "NEEDS_REVIEW", FALLBACK_CONFIDENCE)


def process_dir(input_dir, output_path, jobs: int | None = None) -> None:
    input_dir = Path(input_dir)
    pdf_paths = [str(p) for p in sorted(input_dir.glob("*.pdf"))
                 if CASE_ID_PATTERN.fullmatch(p.stem)]
    # A non-conforming filename stem would make the whole submission invalid,
    # so such files are skipped above.
    jobs = jobs or min(4, cpu_count())
    if jobs <= 1 or len(pdf_paths) <= 1:
        rows = [_process_pdf_str(p) for p in pdf_paths]
    else:
        with Pool(jobs) as pool:
            rows = list(pool.imap_unordered(_process_pdf_str, pdf_paths,
                                            chunksize=8))
    # write_jsonl sorts by case_id, so imap_unordered's arrival order does
    # not affect the output bytes.
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
