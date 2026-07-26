#!/usr/bin/env python3
"""Score resolved records (from tools/run_train.py) against train labels.

Uses the challenge evaluator's normalization rules. Prints per-field accuracy,
evaluator-weighted extraction score, and top mismatch examples.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_LABELS = (Path(__file__).resolve().parent.parent.parent /
                  "mib-doc-challenge" / "data" / "train_labels.csv")

FIELD_WEIGHTS = {
    "applicant_name": 5, "species_code": 6, "home_world": 5, "visa_class": 5,
    "sponsor_id": 5, "arrival_date": 4, "declared_purpose": 3,
    "risk_flags": 8, "fee_status": 4,
}


def normalize(value):
    return " ".join(str(value or "").strip().split()).casefold()


def normalize_flags(value):
    raw = normalize(value)
    if raw in {"", "none", "null", "unknown"}:
        return "none"
    return "|".join(sorted(p.strip() for p in raw.split("|") if p.strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--mismatches", type=int, default=15)
    args = ap.parse_args()

    labels = {r["case_id"]: r for r in csv.DictReader(open(args.labels))}
    records = {}
    with open(args.records) as f:
        for line in f:
            row = json.loads(line)
            records[row["case_id"]] = row

    ok = Counter()
    tot = Counter()
    raw_pts = 0.0
    max_pts = 0.0
    mism = defaultdict(list)

    labels = {cid: lab for cid, lab in labels.items() if cid in records}
    for cid, lab in labels.items():
        rec = records.get(cid)
        fields = (rec or {}).get("fields", {})
        flags = "|".join((rec or {}).get("risk_flags", [])) or "none"
        for field, weight in FIELD_WEIGHTS.items():
            truth = lab[field]
            if field == "risk_flags":
                pred = flags
                match = normalize_flags(truth) == normalize_flags(pred)
            else:
                pred = fields.get(field)
                match = normalize(truth) == normalize(pred)
            tot[field] += 1
            max_pts += weight
            if match:
                ok[field] += 1
                raw_pts += weight
            elif len(mism[field]) < args.mismatches:
                mism[field].append((cid, truth, pred))

    print(f"weighted extraction: {raw_pts:.0f}/{max_pts:.0f} = "
          f"{50 * raw_pts / max_pts:.2f}/50 (vs public-label reference)")
    for field in FIELD_WEIGHTS:
        print(f"  {field:18s} {ok[field]:4d}/{tot[field]}")
    print()
    for field in FIELD_WEIGHTS:
        if mism[field]:
            print(f"-- {field} mismatches (truth | pred):")
            for cid, t, p in mism[field]:
                print(f"   {cid}  {t!r} | {p!r}")


if __name__ == "__main__":
    main()
