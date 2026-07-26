"""Adjudication rule cascade.

Maps a resolved CaseRecord to a rule path: either a deterministic class
distribution (policy rules that hold 100% on the public training labels and
mirror FIELD_MANUAL.md) or an empirical path id whose class distribution is
learned from the public training labels (see tools/fit_stats.py and
mib/data/policy_stats.json).

Provenance: the revoked-sponsor list, embargoed-world list, and staleness
cutoff are policy knowledge fitted on the public train labels — the same
artifact class as model weights. Nothing here is keyed to specific case IDs
or file names; every rule applies to any fresh packet.
"""
from dataclasses import dataclass

# Flags the field manual marks disqualifying (deny), verified 186/186 on train.
DISQUALIFYING_FLAGS = {
    "memory_tampering",
    "planetary_embargo",
    "active_warrant",
    "biohazard_red",
}

# Review-only flags; combinations may escalate, which the empirical paths learn.
REVIEW_FLAGS = {
    "identity_conflict",
    "sponsor_mismatch",
    "illegible_biometrics",
    "rescinded_denial",
}

# Public manual lists SPN-0007/0139/4040; the other three are learned from
# train labels (every non-DIP case with them is DENIED, every DIP-1 approved).
REVOKED_SPONSORS = {
    "SPN-0007", "SPN-0139", "SPN-4040",
    "SPN-7331", "SPN-2718", "SPN-9090",
}

# Every train case from these worlds carries planetary_embargo (32/32, 18/18).
EMBARGOED_WORLDS = {"TRAPPIST-1e", "Eris Relay"}

# 180 days before the dataset version date (2026-07-07). Non-DIP cases with
# arrival before this are DENIED 36/36 on train; DIP-1 is exempt (13/13).
STALE_CUTOFF = "2026-01-08"

_VERDICTS = {"APPROVED", "DENIED", "NEEDS_REVIEW"}
_ONE_HOT = {
    "APPROVED": (1.0, 0.0, 0.0),
    "DENIED": (0.0, 1.0, 0.0),
    "NEEDS_REVIEW": (0.0, 0.0, 1.0),
}


@dataclass
class RulePath:
    path_id: str
    # (pA, pD, pR) for deterministic rules; None means "look up the learned
    # distribution for path_id in policy_stats.json".
    dist: tuple | None


def infer_flags(record) -> set:
    """Risk flags after inference. Currently: embargoed home world implies
    planetary_embargo (deterministic on train)."""
    flags = set(record.risk_flags)
    if record.fields.get("home_world") in EMBARGOED_WORLDS:
        flags.add("planetary_embargo")
    return flags


def path_of(record) -> RulePath:
    """First-match-wins cascade. Order is load-bearing and covered by tests."""
    fields = record.fields
    evidence = record.evidence
    flags = infer_flags(record)
    visa = fields.get("visa_class")
    is_dip = visa == "DIP-1"

    verdict = evidence.note_verdict
    if verdict in _VERDICTS:
        return RulePath(f"note:{verdict}", _ONE_HOT[verdict])

    if flags & DISQUALIFYING_FLAGS:
        return RulePath("disq", _ONE_HOT["DENIED"])

    if visa == "TRANSIT-7":
        return RulePath("transit", _ONE_HOT["DENIED"])

    fee = fields.get("fee_status")
    if fee == "unpaid":
        return RulePath("unpaid", _ONE_HOT["DENIED"])
    if fee == "unknown":
        # An extracted, explicit "unknown" — distinct from a missing value.
        return RulePath("feeunk", _ONE_HOT["NEEDS_REVIEW"])

    if fields.get("sponsor_id") in REVOKED_SPONSORS and not is_dip:
        return RulePath("revoked", _ONE_HOT["DENIED"])

    quality = _quality_suffix(record)

    if fields.get("sponsor_id") is None and not is_dip:
        # A missing sponsor is usually an extraction gap, not a document
        # fact; the learned distribution (split by evidence quality) decides.
        return RulePath(f"nosponsor;{quality}", None)

    arrival = fields.get("arrival_date")
    if "arrival_date" in evidence.hidden_only_fields:
        # Manual policy: a date present only in untrusted hidden text is a
        # review condition on the document itself.
        return RulePath("nodate-hidden", _ONE_HOT["NEEDS_REVIEW"])
    if arrival is None:
        return RulePath(f"nodate;{quality}", None)

    if arrival < STALE_CUTOFF and not is_dip:
        return RulePath("stale", _ONE_HOT["DENIED"])

    flag_part = "|".join(sorted(flags)) if flags else "none"
    fee_part = fee if fee is not None else "missing"
    visa_part = "dip" if is_dip else "std"
    return RulePath(
        f"flags:{flag_part};fee:{fee_part};visa:{visa_part};{quality}", None)


def _quality_suffix(record) -> str:
    """Evidence-quality features for empirical path keys. Splits 'clean
    because verified' from 'clean because we could not read the packet'."""
    evidence = record.evidence
    if evidence.unreadable_pages > 0:
        q = "u"  # at least one page defeated OCR entirely
    elif evidence.ocr_used:
        q = "o"  # some evidence came through OCR
    else:
        q = "t"  # pure text-layer case
    missing = sum(1 for name in _QUALITY_FIELDS if record.fields.get(name) is None)
    m = "m0" if missing == 0 else ("m1" if missing == 1 else "m2")
    return f"q:{q};{m}"


_QUALITY_FIELDS = (
    "applicant_name", "species_code", "home_world", "visa_class",
    "sponsor_id", "arrival_date", "declared_purpose", "fee_status",
)
