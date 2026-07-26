#!/usr/bin/env python3
"""Residual analysis: where do classification points and extraction points go?

Joins resolved records (tools/run_train.py output) with public train labels and
reports:
  1. classification-error buckets by (truth, predicted, rule path)
  2. extraction misses by field x evidence quality
  3. single-field counterfactuals: for each field, if its true value were
     recovered, how many wrong adjudications flip to correct (upper bound on
     the classification value of improving that field's extraction)

Read-only diagnostics over public training data; not part of the runtime.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.decide import decide, load_stats
from mib.rules import infer_flags, path_of
from tools.fit_stats import record_from_row

DEFAULT_LABELS = (Path(__file__).resolve().parent.parent.parent /
                  "mib-doc-challenge" / "data" / "train_labels.csv")

FIELDS = ["applicant_name", "species_code", "home_world", "visa_class",
          "sponsor_id", "arrival_date", "declared_purpose", "fee_status"]


def truth_flags(label_row):
    raw = label_row["risk_flags"]
    return set() if raw == "none" else set(raw.split("|"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    labels = {r["case_id"]: r for r in csv.DictReader(open(args.labels))}
    rows = [json.loads(l) for l in open(args.records)]
    stats = load_stats()

    cls_buckets = Counter()
    samples = defaultdict(list)
    ext_miss = Counter()
    base_wrong = {}

    for row in rows:
        cid = row["case_id"]
        lab = labels[cid]
        rec = record_from_row(row)
        rp = path_of(rec)
        pred, _conf = decide(rp, stats)
        truth = lab["adjudication"]
        if pred != truth:
            key = (truth, pred, rp.path_id.split(";")[0])
            cls_buckets[key] += 1
            if len(samples[key]) < 3:
                samples[key].append(cid)
            base_wrong[cid] = (truth, rp.path_id)
        for f in FIELDS:
            t = " ".join(str(lab[f]).split()).casefold()
            p = " ".join(str(rec.fields.get(f) or "").split()).casefold()
            if t != p:
                q = "u" if row["evidence"]["unreadable_pages"] else (
                    "o" if row["evidence"]["ocr_used"] else "t")
                ext_miss[(f, q, "none" if not rec.fields.get(f) else "wrong")] += 1
        tf = truth_flags(lab)
        pf = infer_flags(record_from_row(row))
        if tf != pf:
            q = "u" if row["evidence"]["unreadable_pages"] else (
                "o" if row["evidence"]["ocr_used"] else "t")
            ext_miss[("risk_flags", q,
                      "missing" if tf - pf else "extra")] += 1

    print(f"wrong adjudications: {len(base_wrong)}/1000")
    print("\n== classification-error buckets (truth -> pred @ path-head) ==")
    for key, n in cls_buckets.most_common(args.top):
        print(f"  {key[0]:>12} -> {key[1]:<13} @ {key[2]:<12} {n:4d}   e.g. {samples[key]}")

    print("\n== extraction misses (field, quality, kind) ==")
    for key, n in ext_miss.most_common(args.top + 8):
        print(f"  {key[0]:<16} q={key[1]} {key[2]:<8} {n:4d}")

    print("\n== single-field counterfactual flips (wrong -> correct) ==")
    flips = Counter()
    for row in rows:
        cid = row["case_id"]
        if cid not in base_wrong:
            continue
        lab = labels[cid]
        truth = lab["adjudication"]
        for f in FIELDS + ["risk_flags"]:
            rec = record_from_row(row)
            if f == "risk_flags":
                rec.risk_flags = truth_flags(lab)
                rec.evidence.observed_flags = truth_flags(lab)
            else:
                rec.fields[f] = lab[f]
            rp = path_of(rec)
            pred, _ = decide(rp, stats)
            if pred == truth:
                flips[f] += 1
    for f, n in flips.most_common():
        print(f"  {f:<16} would fix {n:3d} wrong adjudications (raw pts <= {n*8})")


if __name__ == "__main__":
    main()
