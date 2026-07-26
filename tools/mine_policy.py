#!/usr/bin/env python3
"""Re-derive the learned policy constants in mib/rules.py from public data.

Read-only audit script: prints, from the public training labels, the evidence
behind every learned constant — the six revoked sponsors (with the DIP-1
exemption split), the embargoed home worlds, and the staleness cutoff. Run it
to confirm nothing in mib/rules.py is keyed to specific cases:

    python3 tools/mine_policy.py
"""
import csv
from collections import Counter
from pathlib import Path

LABELS = (Path(__file__).resolve().parent.parent.parent /
          "mib-doc-challenge" / "data" / "train_labels.csv")

DISQ = {"memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red"}
STALE_CUTOFF = "2026-01-08"  # dataset version 2026-07-07 minus 180 days


def flags(row):
    raw = row["risk_flags"]
    return set() if raw == "none" else set(raw.split("|"))


def main():
    rows = list(csv.DictReader(open(LABELS)))

    print("== Revoked sponsors ==")
    print("Sponsors whose non-DIP-1 cases are all DENIED despite clean flags/fee")
    print("(public manual lists SPN-0007/0139/4040 and says more may appear):")
    by_sponsor = {}
    for r in rows:
        by_sponsor.setdefault(r["sponsor_id"], []).append(r)
    for sponsor, cases in sorted(by_sponsor.items()):
        nondip = [r for r in cases if r["visa_class"] != "DIP-1"
                  and not (flags(r) & DISQ) and r["fee_status"] in ("paid", "waived")
                  and r["visa_class"] != "TRANSIT-7"]
        denied = [r for r in nondip if r["adjudication"] == "DENIED"]
        if len(nondip) >= 3 and len(denied) == len(nondip):
            dip = [r for r in cases if r["visa_class"] == "DIP-1"]
            dip_ok = sum(r["adjudication"] == "APPROVED" for r in dip)
            print(f"  {sponsor}: {len(denied)}/{len(nondip)} non-DIP denied; "
                  f"DIP-1 {dip_ok}/{len(dip)} approved")

    print("\n== Embargoed home worlds ==")
    print("Worlds where every case carries planetary_embargo:")
    by_world = {}
    for r in rows:
        by_world.setdefault(r["home_world"], []).append(r)
    for world, cases in sorted(by_world.items()):
        n = sum("planetary_embargo" in flags(r) for r in cases)
        if n == len(cases):
            print(f"  {world}: {n}/{len(cases)}")

    print("\n== Staleness cutoff ==")
    print(f"Non-DIP-1 cases with arrival < {STALE_CUTOFF}, after removing cases "
          "explained by flags/fee/transit rules (stale=True is uniformly DENIED):")
    conf = Counter()
    for r in rows:
        if flags(r) & DISQ or r["visa_class"] == "TRANSIT-7":
            continue
        if r["fee_status"] in ("unpaid", "unknown") or flags(r):
            continue
        stale = r["arrival_date"] < STALE_CUTOFF and r["visa_class"] != "DIP-1"
        conf[(stale, r["adjudication"])] += 1
    for key in sorted(conf):
        print(f"  stale={key[0]!s:5} {key[1]}: {conf[key]}")


if __name__ == "__main__":
    main()
