#!/usr/bin/env python3
"""Run the extraction pipeline over training PDFs and dump resolved records.

Output JSONL rows carry the resolved fields plus evidence needed by
tools/score_extraction.py and tools/fit_stats.py. Adjudication is a
placeholder until mib/rules.py exists (Task 7).
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.extract import extract_document
from mib.resolve import resolve_case

DEFAULT_TRAIN = Path(__file__).resolve().parent.parent.parent / "mib-doc-challenge" / "data" / "train"


def process_one(pdf_path_str):
    pdf_path = Path(pdf_path_str)
    stem = pdf_path.stem
    try:
        doc = extract_document(pdf_path_str)
        record = resolve_case(doc, active_case_id=stem, pdf_path=pdf_path_str)
        fields = dict(record.fields)
        ev = record.evidence
        return {
            # Always the filename stem: it is the packet's active case id and
            # the only id the scorer accepts. Extracted ids stay in fields.
            "case_id": stem,
            "fields": fields,
            "risk_flags": sorted(record.risk_flags),
            "evidence": {
                "note_verdict": ev.note_verdict,
                "note_reason": ev.note_reason,
                "waiver_code": ev.waiver_code,
                "observed_flags": sorted(ev.observed_flags),
                "registry_status": ev.registry_status,
                "conflicts": sorted(ev.conflicts),
                "has_image_only_pages": ev.has_image_only_pages,
                "ocr_used": ev.ocr_used,
                "hidden_only_fields": sorted(ev.hidden_only_fields),
                "unreadable_pages": ev.unreadable_pages,
                "saw_observed_flags": ev.saw_observed_flags,
                "doc_types_seen": sorted(ev.doc_types_seen),
                "bio_confidence": ev.bio_confidence,
                "unknown_content_pages": ev.unknown_content_pages,
                "foreign_pages": ev.foreign_pages,
            },
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - fallback row must always exist
        return {"case_id": stem, "fields": {}, "risk_flags": [],
                "evidence": None, "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(DEFAULT_TRAIN))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pdfs = sorted(str(p) for p in Path(args.input_dir).glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    with Pool(args.jobs) as pool:
        rows = pool.map(process_one, pdfs, chunksize=8)

    rows.sort(key=lambda r: r["case_id"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    errors = [r for r in rows if r["error"]]
    print(f"processed {len(rows)} pdfs, {len(errors)} errors")
    for r in errors[:10]:
        print("  ERROR", r["case_id"], r["error"])


if __name__ == "__main__":
    main()
