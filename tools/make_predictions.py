#!/usr/bin/env python3
"""Convert cached resolved records (tools/run_train.py output) into a
predictions.jsonl, applying rules + decide + output serialization. Lets the
scoring loop iterate without re-OCRing PDFs. mib/main.py is the equivalent
uncached container path."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.decide import decide, load_stats
from mib.output import build_row, write_jsonl
from mib.rules import infer_flags, path_of
from tools.fit_stats import record_from_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = load_stats()
    rows = []
    with open(args.records) as f:
        for line in f:
            raw = json.loads(line)
            case_id = raw["case_id"]
            if raw.get("error"):
                rows.append(build_row(case_id, {}, "NEEDS_REVIEW", 0.3))
                continue
            record = record_from_row(raw)
            flags = infer_flags(record)
            adjudication, confidence = decide(path_of(record), stats)
            fields = dict(record.fields)
            fields["risk_flags"] = "|".join(sorted(flags)) if flags else "none"
            rows.append(build_row(case_id, fields, adjudication, confidence))
    write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
