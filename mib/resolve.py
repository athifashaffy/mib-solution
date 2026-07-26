"""Cross-page precedence merge: DocText -> CaseRecord.

`mib.extract` splits each page into visible/hidden/OCR text buckets, and
`mib.parse` turns one page's text into raw (unsnapped) field evidence. This
module is the only place that:

  - decides which page text gets handed to `parse_page` at all (hidden text
    is *never* parsed -- see `resolve_case`'s docstring for why);
  - snaps raw evidence against the closed vocabularies (`mib.vocab`) or the
    regex-normalized shapes for sponsor_id/arrival_date/case_id;
  - picks a single winning value per field across every page/doc-type/source
    the case contains, by a fixed precedence order;
  - detects the specific cross-doc-type disagreements ("identity_conflict",
    "sponsor_mismatch") that feed risk flags.

Everything here operates on *trusted* evidence only. A field with no
trustworthy evidence resolves to `None` -- never a placeholder. Placeholders
(e.g. "SPN-0000", "1900-01-01") are an output-boundary concern applied only
by `mib.output.build_row`, not here.
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from mib import vocab
from mib.parse import parse_page

# --- data shapes ------------------------------------------------------------

# The nine fields every CaseRecord always has (value or None), in the same
# order the submission schema expects them.
FIELD_NAMES = [
    "case_id",
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "fee_status",
]


@dataclass
class EvidenceSummary:
    note_verdict: str | None = None
    note_reason: str | None = None
    waiver_code: str | None = None
    observed_flags: set = field(default_factory=set)
    registry_status: str | None = None
    conflicts: set = field(default_factory=set)
    has_image_only_pages: bool = False
    ocr_used: bool = False
    hidden_only_fields: set = field(default_factory=set)
    unreadable_pages: int = 0
    # True iff ANY observed-flags evidence line was parsed (even "none").
    # False means the packet's biometric reading is entirely absent, which
    # is a materially different risk situation than a slip that said "none".
    saw_observed_flags: bool = False
    # Page doc types recognized anywhere in the packet (text or OCR).
    doc_types_seen: set = field(default_factory=set)
    # Raw "Biometric confidence" value from the slip (e.g. "86%"), if any.
    bio_confidence: str | None = None
    # Image-only pages whose OCR parsed to no recognizable doc type: content
    # pages destroyed beyond classification (distinct from unreadable_pages,
    # which counts pages whose OCR produced almost no text at all).
    unknown_content_pages: int = 0
    # Pages discarded because their own Case ID belongs to another case.
    foreign_pages: int = 0


@dataclass
class CaseRecord:
    fields: dict
    evidence: EvidenceSummary
    risk_flags: set = field(default_factory=set)


# --- doc-type / source precedence -------------------------------------------

# Lower number = higher precedence. Manual corrections aren't a doc type;
# they're applied as an override before this ranking is ever consulted.
_DOC_TYPE_RANK = {
    "intake": 1,
    "biometric": 2,
    "sponsor": 3,
    "registry": 4,
    "fee": 5,
}
_UNRANKED_RANK = 99

# text beats ocr at equal doc-type rank.
_SOURCE_RANK = {"text": 0, "ocr": 1}

# Which parse.py field-dict keys feed each resolved CaseRecord field. Most
# fields have a single source key; applicant_name/visa_class/sponsor_id also
# accept the sponsor-attestation page's separately-keyed evidence
# (attest_name/attest_visa/attest_sponsor), which already carries doc_type
# "sponsor" from parse_page's own doc-type detection.
_FIELD_SOURCE_KEYS = {
    "case_id": ["case_id"],
    "applicant_name": ["applicant_name", "attest_name"],
    "species_code": ["species_code"],
    "home_world": ["home_world"],
    "visa_class": ["visa_class", "attest_visa"],
    "sponsor_id": ["sponsor_id", "attest_sponsor"],
    "arrival_date": ["arrival_date"],
    "declared_purpose": ["declared_purpose"],
    "fee_status": ["fee_status"],
}


# --- value snapping ----------------------------------------------------------

# OCR digit/letter confusions this dataset exhibits, restricted to the
# digit-shaped tail of a value (never applied to literal prefixes like
# "SPN-"/"MIB-", which are matched separately). "S"->"5" only matters for
# sponsor IDs; case IDs/dates never contain a legitimate "S", but excluding
# it there is simply more conservative.
_DIGIT_FIX_WITH_S = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "s": "5"})
_DIGIT_FIX_NO_S = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"})

_SPONSOR_SHAPE_RE = re.compile(r"(?i)^SPN-(.+)$")
_CASE_ID_SHAPE_RE = re.compile(r"(?i)^MIB-(.+)$")
_DATE_SHAPE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalize_sponsor(raw) -> str | None:
    if not raw:
        return None
    value = raw.strip().replace(" ", "")
    m = _SPONSOR_SHAPE_RE.match(value)
    if not m:
        return None
    digits = m.group(1).translate(_DIGIT_FIX_WITH_S)
    candidate = f"SPN-{digits}"
    return candidate if re.fullmatch(r"SPN-\d{4}", candidate) else None


def _normalize_case_id(raw) -> str | None:
    if not raw:
        return None
    value = raw.strip().replace(" ", "")
    m = _CASE_ID_SHAPE_RE.match(value)
    if not m:
        return None
    digits = m.group(1).translate(_DIGIT_FIX_NO_S)
    candidate = f"MIB-{digits}"
    return candidate if re.fullmatch(r"MIB-\d{6}", candidate) else None


def _normalize_date(raw) -> str | None:
    if not raw:
        return None
    value = raw.strip().replace(" ", "")
    fixed = value.translate(_DIGIT_FIX_NO_S)
    if not _DATE_SHAPE_RE.match(fixed):
        return None
    try:
        date.fromisoformat(fixed)
    except ValueError:
        return None
    return fixed


def _snap_field(field_name: str, raw: str) -> str | None:
    """Snap a raw field value to its canonical form for `field_name`, or
    None if it doesn't snap cleanly."""
    if field_name == "species_code":
        return vocab.snap(raw, "species")
    if field_name == "home_world":
        return vocab.snap(raw, "world")
    if field_name == "visa_class":
        return vocab.snap(raw, "visa")
    if field_name == "declared_purpose":
        return vocab.snap(raw, "purpose")
    if field_name == "fee_status":
        return vocab.snap(raw, "fee")
    if field_name == "applicant_name":
        return vocab.snap_name(raw)
    if field_name == "sponsor_id":
        return _normalize_sponsor(raw)
    if field_name == "arrival_date":
        return _normalize_date(raw)
    if field_name == "case_id":
        return _normalize_case_id(raw)
    return None


# --- hidden-text plausibility detection (for hidden_only_fields) -----------

_HIDDEN_PATTERNS = {
    "sponsor_id": re.compile(r"SPN-\d{4}"),
    "arrival_date": re.compile(r"\d{4}-\d{2}-\d{2}"),
    "case_id": re.compile(r"MIB-\d{6}"),
}


def _is_foreign_page(page_parse, active_case_id: str | None) -> bool:
    """A page is foreign iff its own cleanly-read Case ID differs from the
    packet's active case id. Only an EXACT clean read counts (same rule as
    conflict detection): an OCR-garbled id must not disqualify a page. The
    packet footer ("Packet MIB-xxxxxx / page N") always carries the active
    id, so only the page's labeled Case ID field is consulted."""
    if not active_case_id:
        return False
    for raw, _dt, _source in page_parse.fields.get("case_id", []):
        normalized = _normalize_case_id(raw)
        if (normalized is not None and _is_exact_read(raw, normalized)
                and normalized != active_case_id):
            return True
    return False


def _hidden_only_fields(doc, resolved_fields: dict) -> set:
    hidden_text = "\n".join(pg.hidden for pg in doc.pages if pg.hidden)
    result = set()
    if not hidden_text:
        return result
    for field_name, pattern in _HIDDEN_PATTERNS.items():
        if resolved_fields.get(field_name) is None and pattern.search(hidden_text):
            result.add(field_name)
    return result


# --- per-field value resolution ---------------------------------------------


def _rank_for(doc_type: str) -> int:
    return _DOC_TYPE_RANK.get(doc_type, _UNRANKED_RANK)


def _resolve_field(field_name: str, all_fields: dict, corrections: list) -> str | None:
    # Manual correction outranks everything else -- but only if it actually
    # snaps; an unsnappable correction is discarded rather than blanking an
    # otherwise-good field (falls through to normal precedence below).
    for raw in corrections:
        snapped = _snap_field(field_name, raw)
        if snapped is not None:
            return snapped

    candidates = []
    for key in _FIELD_SOURCE_KEYS.get(field_name, []):
        for raw, doc_type, source in all_fields.get(key, []):
            snapped = _snap_field(field_name, raw)
            if snapped is None:
                continue
            candidates.append((_rank_for(doc_type), _SOURCE_RANK.get(source, 1), snapped))

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c[0], c[1]))
    best_rank, best_source = candidates[0][0], candidates[0][1]
    # Ties at the best (rank, source) tier resolve to the first-seen value
    # (stable sort preserves original page-encounter order).
    for rank, source, value in candidates:
        if rank == best_rank and source == best_source:
            return value
    return None  # unreachable


# --- conflict detection ------------------------------------------------------


def _first_value_for_doc_type(field_key: str, all_fields: dict, doc_type: str, snap_fn) -> str | None:
    """First cleanly-read value of `field_key` from pages of `doc_type`.

    Conflict detection intentionally accepts only EXACT reads (the snapped
    value equals the raw text up to whitespace/case): planted conflicts in
    these documents are printed cleanly on purpose, while an OCR-garbled
    name that fuzzy-snaps to a *different* vocabulary entry would fabricate
    a phantom conflict. Requiring exactness keeps conflict evidence precise
    without weakening genuine-conflict recall."""
    for raw, dt, _source in all_fields.get(field_key, []):
        if dt != doc_type:
            continue
        snapped = snap_fn(raw)
        if snapped is not None and _is_exact_read(raw, snapped):
            return snapped
    return None


def _is_exact_read(raw: str, snapped: str) -> bool:
    return " ".join(raw.split()).casefold() == " ".join(snapped.split()).casefold()


def _detect_conflicts(all_fields: dict) -> set:
    conflicts = set()

    intake_name = _first_value_for_doc_type("applicant_name", all_fields, "intake", vocab.snap_name)
    registry_name = _first_value_for_doc_type("applicant_name", all_fields, "registry", vocab.snap_name)
    biometric_name = _first_value_for_doc_type("applicant_name", all_fields, "biometric", vocab.snap_name)
    if intake_name is not None and registry_name is not None and intake_name != registry_name:
        conflicts.add("identity_conflict")
    if intake_name is not None and biometric_name is not None and intake_name != biometric_name:
        conflicts.add("identity_conflict")

    intake_sponsor = _first_value_for_doc_type("sponsor_id", all_fields, "intake", _normalize_sponsor)
    attest_sponsor = _first_value_for_doc_type("attest_sponsor", all_fields, "sponsor", _normalize_sponsor)
    if intake_sponsor is not None and attest_sponsor is not None and intake_sponsor != attest_sponsor:
        conflicts.add("sponsor_mismatch")

    return conflicts


# --- evidence-summary scalar fields (text-over-ocr, disagreement -> None) --


def _resolve_text_over_ocr(evidences: list) -> str | None:
    """Text-source evidence wins outright over ocr-source evidence; within
    whichever source group is chosen, unanimous agreement returns the value
    and any disagreement returns None (rather than guessing)."""
    text_vals = [v for v, _dt, source in evidences if source == "text"]
    ocr_vals = [v for v, _dt, source in evidences if source == "ocr"]
    for vals in (text_vals, ocr_vals):
        if not vals:
            continue
        uniq = set(vals)
        return vals[0] if len(uniq) == 1 else None
    return None


# --- observed_flags -----------------------------------------------------------

_FLAG_SPLIT_RE = re.compile(r"[,|]")


def _resolve_observed_flags(evidences: list) -> set:
    flags = set()
    for raw, _dt, _source in evidences:
        for token in _FLAG_SPLIT_RE.split(raw):
            token = token.strip()
            if not token:
                continue
            snapped = vocab.snap(token, "flag")
            if snapped is not None:
                flags.add(snapped)
                continue
            # OCR often appends garbage to a real flag on the same
            # comma-separated part ("biohazard_red $b ane Yo"). Salvage by
            # snapping each whitespace word individually; short noise words
            # can't reach any flag within the snap threshold, so this never
            # invents a flag that isn't closely present.
            for word in token.split():
                word_snapped = vocab.snap(word, "flag")
                if word_snapped is not None:
                    flags.add(word_snapped)
    return flags


# --- unreadable page detection ------------------------------------------------

_UNREADABLE_OCR_MIN_CHARS = 50


def _is_unreadable_ocr(ocr_text: str | None) -> bool:
    if not ocr_text:
        return True
    non_ws = sum(1 for c in ocr_text if not c.isspace())
    return non_ws < _UNREADABLE_OCR_MIN_CHARS


# --- entry point ---------------------------------------------------------------


_SWEEP_SPONSOR_RE = re.compile(r"\bSPN-\d{4}\b")
_SWEEP_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
# Plausible arrival window for this program (train range is 2025-05..2026-07;
# a generous pad keeps the sweep from adopting OCR junk like 2109-91-14).
_SWEEP_DATE_MIN = "2025-01-01"
_SWEEP_DATE_MAX = "2026-12-31"


def _pattern_sweep(trusted_texts: list, resolved_fields: dict) -> dict:
    """Fill still-None sponsor_id/arrival_date from bare pattern hits in
    trusted, non-foreign page text. Most-frequent hit wins; a tie between
    distinct values means ambiguity, so nothing is filled."""
    wanted = [n for n in ("sponsor_id", "arrival_date")
              if resolved_fields.get(n) is None]
    if not wanted:
        return {}

    text = "\n".join(trusted_texts)

    out: dict = {}
    if "sponsor_id" in wanted:
        hits = Counter(_SWEEP_SPONSOR_RE.findall(text))
        if hits:
            (top, n), *rest = hits.most_common()
            if not rest or rest[0][1] < n:
                out["sponsor_id"] = top
    if "arrival_date" in wanted:
        dates = [d for d in _SWEEP_DATE_RE.findall(text)
                 if _SWEEP_DATE_MIN <= d <= _SWEEP_DATE_MAX and _is_real_date(d)]
        hits = Counter(dates)
        if hits:
            (top, n), *rest = hits.most_common()
            if not rest or rest[0][1] < n:
                out["arrival_date"] = top
    return out


def _is_real_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def resolve_case(doc, active_case_id: str | None = None) -> CaseRecord:
    """Merge every page of `doc` into a single CaseRecord.

    Per page: `pg.visible` is always parsed with source="text". If the page
    is `image_only` and has OCR text, that OCR text is *additionally* parsed
    with source="ocr" (OCR of a page with a real text layer is never
    parsed). `pg.hidden` is never handed to `parse_page` under any
    circumstance -- hidden spans are a prompt-injection vector, not a value
    source; the only thing this module does with hidden text is check it
    against a few plausibility regexes for `hidden_only_fields` reporting.

    `active_case_id` is the packet's own case id (the PDF filename stem).
    A packet can contain pages for more than one applicant; policy says to
    use the applicant attached to the active case id. Any page whose own
    cleanly-read Case ID differs from `active_case_id` is a foreign page:
    ALL of its evidence is discarded.
    """
    has_image_only = False
    ocr_used = False
    unreadable_pages = 0
    unknown_content_pages = 0
    doc_types_seen: set = set()
    foreign_pages = 0

    all_fields: dict[str, list] = {}
    corrections: dict[str, list] = {}
    trusted_texts: list = []

    for pg in doc.pages:
        if pg.image_only:
            has_image_only = True
            if _is_unreadable_ocr(pg.ocr):
                unreadable_pages += 1

        text_parse = parse_page(pg.visible, source="text")
        if _is_foreign_page(text_parse, active_case_id):
            foreign_pages += 1
        else:
            trusted_texts.append(pg.visible)
            if text_parse.doc_type not in ("cover", "unknown"):
                doc_types_seen.add(text_parse.doc_type)
            for key, evs in text_parse.fields.items():
                all_fields.setdefault(key, []).extend(evs)
            for key, value in text_parse.corrections.items():
                corrections.setdefault(key, []).append(value)

        if pg.image_only and pg.ocr:
            ocr_used = True
            ocr_parse = parse_page(pg.ocr, source="ocr")
            if _is_foreign_page(ocr_parse, active_case_id):
                foreign_pages += 1
            else:
                trusted_texts.append(pg.ocr)
                if ocr_parse.doc_type not in ("cover", "unknown"):
                    doc_types_seen.add(ocr_parse.doc_type)
                for key, evs in ocr_parse.fields.items():
                    all_fields.setdefault(key, []).extend(evs)
                for key, value in ocr_parse.corrections.items():
                    corrections.setdefault(key, []).append(value)

    resolved_fields = {
        name: _resolve_field(name, all_fields, corrections.get(name, []))
        for name in FIELD_NAMES
    }

    # Last-resort pattern sweep: sponsor ids and arrival dates also appear in
    # attestation prose and registry margins that the label-anchored parsers
    # miss when OCR mangles the labels. Only fills a field that resolved to
    # None, only from TRUSTED, non-foreign page text (visible layers + OCR
    # of image-only pages, never hidden spans, never pages belonging to a
    # different applicant's case id), and only when the pattern hits are
    # unambiguous enough to take the most frequent one.
    for name, value in _pattern_sweep(trusted_texts, resolved_fields).items():
        resolved_fields[name] = value

    conflicts = _detect_conflicts(all_fields)
    observed_flags = _resolve_observed_flags(all_fields.get("observed_flags", []))

    registry_status_raw = _resolve_text_over_ocr(all_fields.get("registry_status", []))
    registry_status = registry_status_raw.strip().upper() if registry_status_raw else None

    evidence = EvidenceSummary(
        note_verdict=_resolve_text_over_ocr(all_fields.get("note_verdict", [])),
        note_reason=_resolve_text_over_ocr(all_fields.get("note_reason", [])),
        waiver_code=_resolve_text_over_ocr(all_fields.get("waiver_code", [])),
        observed_flags=observed_flags,
        registry_status=registry_status,
        conflicts=conflicts,
        has_image_only_pages=has_image_only,
        ocr_used=ocr_used,
        hidden_only_fields=_hidden_only_fields(doc, resolved_fields),
        unreadable_pages=unreadable_pages,
        saw_observed_flags=bool(all_fields.get("observed_flags")),
        doc_types_seen=doc_types_seen,
        bio_confidence=_resolve_text_over_ocr(all_fields.get("bio_confidence", [])),
        foreign_pages=foreign_pages,
    )

    risk_flags = observed_flags | conflicts

    return CaseRecord(fields=resolved_fields, evidence=evidence, risk_flags=risk_flags)
