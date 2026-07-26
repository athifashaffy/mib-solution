#!/usr/bin/env python3
"""Fit per-path class distributions and out-of-fold decision accuracies.

Reads resolved records (tools/run_train.py output) plus the public train
labels, groups truth adjudications by rules.path_of path_id, and writes
mib/data/policy_stats.json:

    {path_id: {"dist": [pA, pD, pR], "n": int,
               "decided_label_cv_accuracy": float | null}}

- dist is Laplace-smoothed (alpha=0.5) over the full train set.
- decided_label_cv_accuracy is measured OUT-OF-FOLD (5 folds assigned by
  sorted case_id index): fold k's cases are decided using distributions
  fitted on the other four folds, so the stored accuracy — used as the
  emitted confidence — is leakage-free.

Provenance: fitted on the public training labels only; keyed by rule-path
structure, never by case id. Deterministic: sorted keys, fixed fold split.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.decide import FA_GUARD_MAX_PD, expected_points
from mib.resolve import CaseRecord, EvidenceSummary
from mib.rules import path_of

DEFAULT_LABELS = (Path(__file__).resolve().parent.parent.parent /
                  "mib-doc-challenge" / "data" / "train_labels.csv")
OUT_PATH = Path(__file__).resolve().parent.parent / "mib" / "data" / "policy_stats.json"

ALPHA = 0.5
FOLDS = 5
LABELS3 = ("APPROVED", "DENIED", "NEEDS_REVIEW")


def record_from_row(row) -> CaseRecord:
    ev = row.get("evidence") or {}
    evidence = EvidenceSummary(
        note_verdict=ev.get("note_verdict"),
        note_reason=ev.get("note_reason"),
        waiver_code=ev.get("waiver_code"),
        observed_flags=set(ev.get("observed_flags", [])),
        registry_status=ev.get("registry_status"),
        conflicts=set(ev.get("conflicts", [])),
        has_image_only_pages=ev.get("has_image_only_pages", False),
        ocr_used=ev.get("ocr_used", False),
        hidden_only_fields=set(ev.get("hidden_only_fields", [])),
        unreadable_pages=ev.get("unreadable_pages", 0),
        saw_observed_flags=ev.get("saw_observed_flags", False),
        doc_types_seen=set(ev.get("doc_types_seen", [])),
        bio_confidence=ev.get("bio_confidence"),
        unknown_content_pages=ev.get("unknown_content_pages", 0),
        foreign_pages=ev.get("foreign_pages", 0),
    )
    return CaseRecord(fields=row.get("fields") or {}, evidence=evidence,
                      risk_flags=set(row.get("risk_flags", [])))


def smoothed_dist(counter: Counter) -> list:
    n = sum(counter.values())
    return [(counter[l] + ALPHA) / (n + 3 * ALPHA) for l in LABELS3]


def decide_label(dist) -> str:
    scores = expected_points(*dist)
    best = max(scores)
    for idx in (2, 1, 0):
        if scores[idx] == best:
            label = LABELS3[idx]
            break
    else:
        raise AssertionError
    # Mirror mib.decide's catastrophic-false-approval guard so the stored
    # out-of-fold accuracies describe the decisions actually emitted.
    if label == "APPROVED" and dist[1] > FA_GUARD_MAX_PD:
        label = "NEEDS_REVIEW"
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True,
                    help="JSONL from tools/run_train.py over data/train")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    labels = {r["case_id"]: r["adjudication"]
              for r in csv.DictReader(open(args.labels))}

    cases = []  # (case_id, path, truth)
    with open(args.records) as f:
        for line in f:
            row = json.loads(line)
            cid = row["case_id"]
            if cid not in labels:
                continue
            rp = path_of(record_from_row(row))
            cases.append((cid, rp, labels[cid]))
    cases.sort(key=lambda c: c[0])

    # Full-train distributions (used at inference for empirical paths).
    by_path = defaultdict(Counter)
    for _, rp, truth in cases:
        by_path[rp.path_id][truth] += 1

    # Out-of-fold decision correctness per path.
    fold_of = {cid: i % FOLDS for i, (cid, _, _) in enumerate(cases)}
    oof_correct = defaultdict(int)
    oof_total = defaultdict(int)
    for k in range(FOLDS):
        train_counts = defaultdict(Counter)
        for cid, rp, truth in cases:
            if fold_of[cid] != k:
                train_counts[rp.path_id][truth] += 1
        for cid, rp, truth in cases:
            if fold_of[cid] != k:
                continue
            if rp.dist is not None:
                decided = decide_label(rp.dist)
            else:
                decided = decide_label(smoothed_dist(train_counts[rp.path_id]))
            oof_total[rp.path_id] += 1
            oof_correct[rp.path_id] += int(decided == truth)

    stats = {}
    for path_id in sorted(by_path):
        n = sum(by_path[path_id].values())
        acc = (oof_correct[path_id] / oof_total[path_id]
               if oof_total[path_id] else None)
        stats[path_id] = {
            "dist": [round(x, 6) for x in smoothed_dist(by_path[path_id])],
            "n": n,
            "decided_label_cv_accuracy": round(acc, 6) if acc is not None else None,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} with {len(stats)} paths over {len(cases)} cases")
    for path_id, entry in sorted(stats.items(), key=lambda kv: -kv[1]["n"])[:20]:
        print(f"  {path_id:55s} n={entry['n']:4d} dist={entry['dist']} "
              f"cv_acc={entry['decided_label_cv_accuracy']}")


if __name__ == "__main__":
    main()
