"""Per-page doc-type detection and field/evidence parsing.

Each PDF page's text (already separated into visible/hidden/OCR buckets by
`mib.extract`) is parsed independently by `parse_page(text, source)`. This
module never looks at hidden text or decides precedence across pages -- it
just recognizes, on a single page's text, which document type the page is
and what raw field values it carries as `Evidence` tuples. Cross-page
precedence, snapping against the closed vocabularies, and conflict
detection are Task 5's job (`mib.resolve`); this module hands it raw
material.

Two on-page layouts are handled:
  - Text-layer forms (intake/fee/registry) put a field's label and its
    value on separate, consecutive lines with no separator, e.g.
    "Case ID\\nMIB-000912".
  - Biometric slips and OCR'd (tesseract) pages put label and value on the
    same line separated by a colon (or an OCR-mangled "|"), e.g.
    "Case ID: MIB-000912". OCR also mangles individual label characters
    (0/O, 1/l confusion), which label matching tolerates; VALUES are never
    "corrected" here -- they're captured raw and cleaned up in resolve().

Adjudicator verdicts are adversarial territory: a note page shows the
verdict as a bare word ("DENIED") *and* as a sentence ("Finding: DENIED.
Reason: ..."), and other pages carry decoy strings ("SAMPLE DENIAL",
"COPY ARTIFACT", "CASEWORK") that look like verdict evidence but are not.
Only the "Finding:" sentence is ever trusted for `note_verdict`/
`note_reason`. Scanned (OCR'd) note pages routinely mangle this sentence's
punctuation ("Finding;" for "Finding:", no punctuation before "Reason",
"Reason;" for "Reason:", digit/letter noise in the verdict word itself
like "APPR0VED") -- the Finding regex tolerates all of that, and a verdict
word that isn't an exact vocabulary match is snapped via Levenshtein
distance (<=1, must be unique) rather than dropped outright.
"""
import re
from dataclasses import dataclass, field

# (value_raw, doc_type, source) -- doc_type is the page's own doc type
# (used by resolve.py for cross-doc-type precedence); source is whichever
# text bucket the caller extracted this text from ("text" or "ocr").
Evidence = tuple[str, str, str]


@dataclass
class PageParse:
    doc_type: str
    fields: dict = field(default_factory=dict)
    corrections: dict = field(default_factory=dict)


# --- doc-type detection -------------------------------------------------

# Checked in this order against the lower-cased first few lines of page
# text; first match wins. Order doesn't matter for correctness here since
# the phrases are mutually exclusive in practice, but is kept stable.
_DOC_TYPE_KEYWORDS = [
    ("form i-8090", "intake"),
    ("mib fee receipt", "fee"),
    ("biometric scan slip", "biometric"),
    ("form b-13", "biometric"),
    ("sponsor attestation", "sponsor"),
    ("planetary registry extract", "registry"),
    ("manual adjudicator note", "note"),
]

# Doc-type header keywords only need to be recognized near the top of the
# page; scanning the whole page risks false positives from later prose.
_HEADER_SCAN_LINES = 5

# Boilerplate that appears on essentially every page regardless of doc
# type: the packet/page footer, the "Synthetic hiring challenge document"
# marker, and the "<CASE-ID> | MIB Eyes Only" header line. A page whose
# every non-blank line is one of these (no header keyword, no fields) is a
# blank filler/cover page rather than a genuinely unrecognizable one.
_BOILERPLATE_LINE_RE = re.compile(
    r"(?i)^("
    r"packet\s+mib-\d+\s*/\s*page\s+\d+"
    r"|synthetic hiring challenge document\.?"
    r"|mib-\d+\s*\|\s*mib eyes only"
    r")$"
)


# Header phrases eligible for tokenwise fuzzy matching on OCR'd pages
# (e.g. "Manual Adjudicater Note"). Phrases containing form numbers are
# excluded: their digit/dash tokens degrade into arbitrary garbage under
# OCR, so fuzzy-matching them would invite false positives.
_FUZZY_DOC_TYPE_PHRASES = [
    ("mib fee receipt", "fee"),
    ("biometric scan slip", "biometric"),
    ("sponsor attestation", "sponsor"),
    ("planetary registry extract", "registry"),
    ("manual adjudicator note", "note"),
]


def _word_allowed_distance(word: str) -> int:
    if len(word) < 4:
        return 0
    if len(word) < 8:
        return 1
    return 2


def _fuzzy_phrase_in_words(phrase_words: list[str], header_words: list[str]) -> bool:
    span = len(phrase_words)
    for start in range(len(header_words) - span + 1):
        window = header_words[start:start + span]
        if all(
            _levenshtein(w, pw) <= _word_allowed_distance(pw)
            for w, pw in zip(window, phrase_words)
        ):
            return True
    return False


def _detect_doc_type(lines: list[str]) -> str:
    header = "\n".join(lines[:_HEADER_SCAN_LINES]).lower()
    for needle, doc_type in _DOC_TYPE_KEYWORDS:
        if needle in header:
            return doc_type
    header_words = re.findall(r"[a-z]+", header)
    for phrase, doc_type in _FUZZY_DOC_TYPE_PHRASES:
        if _fuzzy_phrase_in_words(phrase.split(), header_words):
            return doc_type
    for line in lines:
        stripped = line.strip()
        if stripped and not _BOILERPLATE_LINE_RE.match(stripped):
            return "unknown"
    return "cover"


# --- value cleanup -------------------------------------------------------

# Values this long are presumed to be OCR garbage/prose bleed rather than a
# genuine short field value (case IDs, names, codes, dates all fit well
# under this). Fields that are legitimately long free text (observed_flags
# comma-lists, note_reason sentences) opt out via long_ok=True.
_MAX_VALUE_LENGTH = 40

# Damage markers meaning "the underlying value is unreadable/absent" often
# survive OCR as (or get misread into) these substrings regardless of the
# original bracketed marker text (e.g. "[VISA CLASS TORN]", or OCR noise
# that reads a damaged region as "REDACTED?"). A captured value containing
# either -- case-insensitively -- is treated as not captured at all.
_DAMAGE_SUBSTRINGS = ("redacted", "torn")


def _clean_value(raw, *, long_ok: bool = False) -> str | None:
    """Raw-capture cleanup: strip incidental whitespace/pipe artifacts,
    then drop obvious non-values (empty, too long, or a damage marker).
    Anything else is returned untouched -- no case-folding, no vocabulary
    snapping; that's resolve()'s job."""
    if not raw:
        return None
    value = raw.strip().strip("|").strip()
    if not value:
        return None
    if not long_ok and len(value) >= _MAX_VALUE_LENGTH:
        return None
    lowered = value.lower()
    if any(marker in lowered for marker in _DAMAGE_SUBSTRINGS):
        return None
    return value


# --- label/value field parsing -------------------------------------------

# label text -> (field name, whether long values are acceptable for it).
# "Applicant" (intake/biometric) and "Registry Name" (registry extract) are
# different labels for the same field: the identity of the person the page
# is about.
_LABELS = [
    ("case id", "case_id", False),
    ("applicant", "applicant_name", False),
    ("species code", "species_code", False),
    ("species match", "species_code", False),
    ("home world", "home_world", False),
    ("visa class", "visa_class", False),
    ("sponsor id", "sponsor_id", False),
    ("arrival date", "arrival_date", False),
    ("declared purpose", "declared_purpose", False),
    ("fee status", "fee_status", False),
    ("waiver code", "waiver_code", False),
    ("registry name", "applicant_name", False),
    ("registry status", "registry_status", False),
    ("observed flags", "observed_flags", True),
    ("biometric confidence", "bio_confidence", False),
]


def _fuzzy_label_fragment(label: str) -> str:
    """Build a regex fragment for `label` that tolerates the OCR digit/
    letter confusions this dataset exhibits in label text (e.g. "Sp0nsor
    ID" for "Sponsor ID"). Only the label is fuzzed -- captured values are
    never altered."""
    parts = []
    for ch in label:
        if ch == " ":
            parts.append(r"\s+")
        elif ch == "o":
            parts.append("[oO0]")
        elif ch == "l":
            parts.append("[lI1]")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def _compile_labels():
    compiled = []
    for label, field_name, long_ok in _LABELS:
        frag = _fuzzy_label_fragment(label)
        same_line = re.compile(rf"(?i)^\s*\|?\s*{frag}\s*[:|]\s*(\S.*)$")
        alone = re.compile(rf"(?i)^\s*\|?\s*{frag}\s*[:|]?\s*$")
        compiled.append((same_line, alone, field_name, long_ok))
    return compiled


_COMPILED_LABELS = _compile_labels()


def _parse_labeled_fields(lines: list[str], add) -> None:
    stripped = [ln.strip() for ln in lines]
    n = len(stripped)
    for i, line in enumerate(stripped):
        if not line:
            continue
        for same_line_re, alone_re, field_name, long_ok in _COMPILED_LABELS:
            m = same_line_re.match(line)
            if m:
                add(field_name, m.group(1), long_ok=long_ok)
                break
            if alone_re.match(line):
                j = i + 1
                while j < n and not stripped[j]:
                    j += 1
                if j < n:
                    add(field_name, stripped[j], long_ok=long_ok)
                break


# --- free-text (prose) parsing -------------------------------------------

# Word-wrapped prose (the adjudicator note's "Reason:" sentence, the
# sponsor attestation letter's body) can break a sentence across lines,
# occasionally mid-word with a hyphen. Flattening dehyphenates first, then
# collapses all remaining whitespace (including newlines) to single spaces
# so the prose regexes below can match regardless of how the page wrapped.
def _flatten(text: str) -> str:
    flat = text.replace("-\n", "")
    flat = re.sub(r"\s+", " ", flat)
    return flat.strip()


# Only ever trust the "Finding: <verdict>. Reason: <reason>." sentence for
# adjudication evidence. Note pages also print the verdict as a bare word
# above the Finding line, and unrelated pages carry decoy strings like
# "SAMPLE DENIAL" / "COPY ARTIFACT" / "CASEWORK" -- none of that is
# matched here on purpose.
#
# Scanned (OCR'd) note pages routinely mangle the punctuation around this
# sentence -- "Finding;" instead of "Finding:", no punctuation at all
# between the verdict and "Reason", "Reason;" instead of "Reason:" -- so
# both label colons are tolerated as ":" or ";" and the punctuation between
# the verdict and "Reason" is optional. The verdict capture also tolerates
# stray digits (OCR 0/O confusion, e.g. "APPR0VED"); `_normalize_verdict`
# is responsible for snapping that back to a real vocabulary word (or
# rejecting it).
_FINDING_RE = re.compile(
    r"(?i)Finding[:;]\s*([A-Za-z0-9_ ]+?)[.,;]?\s*Reason[:;]\s*(.+?)"
    r"(?=\s+Packet\s+MIB-|\s+Synthetic\s+hiring|$)"
)

# The only verdicts `_normalize_verdict` will ever emit.
_VALID_VERDICTS = ("APPROVED", "DENIED", "NEEDS_REVIEW")

# Beyond this edit distance a captured verdict word is presumed to be
# unrelated noise rather than an OCR-mangled real verdict -- e.g. "NEMED"
# (distance 4 from DENIED) must never snap to DENIED.
_VERDICT_SNAP_MAX_DISTANCE = 1


def _levenshtein(a: str, b: str) -> int:
    """Standard dynamic-programming edit distance (insert/delete/substitute,
    unit cost). Small/local on purpose -- only ever called against the
    three-entry `_VALID_VERDICTS` tuple, so no need to share `mib.vocab`'s
    (private) implementation."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur_row[j] = min(
                prev_row[j] + 1,
                cur_row[j - 1] + 1,
                prev_row[j - 1] + cost,
            )
        prev_row = cur_row
    return prev_row[-1]


def _normalize_verdict(raw: str) -> str | None:
    value = raw.strip().strip(".,;").strip()
    if not value:
        return None
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_]", "", value)
    if not value:
        return None
    value = value.upper()
    if value in _VALID_VERDICTS:
        return value
    # Not an exact vocabulary member (e.g. OCR noise like "APPR0VED") --
    # try snapping to the closest verdict, but only ever on an unambiguous,
    # close (<=1 edit) match. Ties or distant candidates yield no verdict
    # at all rather than a guess.
    best_distance = None
    best_candidates: list[str] = []
    for candidate in _VALID_VERDICTS:
        distance = _levenshtein(value, candidate)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_candidates = [candidate]
        elif distance == best_distance:
            best_candidates.append(candidate)
    if (
        best_distance is not None
        and best_distance <= _VERDICT_SNAP_MAX_DISTANCE
        and len(best_candidates) == 1
    ):
        return best_candidates[0]
    return None


# OCR can mangle the "Finding"/"Reason" label words themselves ("Firing:",
# "Fnding;", "Reasen:"). Canonicalize any word-plus-colon/semicolon token
# within edit distance 2 of a label word back to the canonical label before
# the Finding regex runs. Only these two label words are ever rewritten,
# and only when directly followed by ':'/';' — the verdict/reason VALUES
# are never altered here, and a fabricated verdict still cannot pass
# `_normalize_verdict`'s distance-<=1 unique-snap gate.
_LABEL_TOKEN_RE = re.compile(r"\b([A-Za-z]{5,9})([:;])")
_FUZZY_LABEL_MAX_DISTANCE = 2


def _canonicalize_note_labels(flat: str) -> str:
    def repl(m: re.Match) -> str:
        word = m.group(1).lower()
        if _levenshtein(word, "finding") <= _FUZZY_LABEL_MAX_DISTANCE:
            return "Finding:"
        if _levenshtein(word, "reason") <= _FUZZY_LABEL_MAX_DISTANCE:
            return "Reason:"
        return m.group(0)

    return _LABEL_TOKEN_RE.sub(repl, flat)


def has_finding_sentence(text: str) -> bool:
    """True iff the page text carries a parseable Finding sentence with a
    valid (snappable) verdict. Used by mib.extract to decide whether a
    higher-DPI OCR retry recovered an adjudicator note."""
    flat = _canonicalize_note_labels(_flatten(text or ""))
    m = _FINDING_RE.search(flat)
    return bool(m) and _normalize_verdict(m.group(1)) is not None


def looks_like_note(text: str) -> bool:
    """Cheap hint that OCR text is probably a Manual Adjudicator Note page
    (mangled or not): the 'djudicat' stem survives most OCR damage, or a
    Finding-like label token is present."""
    if not text:
        return False
    lowered = text.lower()
    if "djudicat" in lowered:
        return True
    for m in _LABEL_TOKEN_RE.finditer(text):
        if _levenshtein(m.group(1).lower(), "finding") <= _FUZZY_LABEL_MAX_DISTANCE:
            return True
    return False


def _fuzzy_word_present(text: str, target: str, max_distance: int = 1) -> bool:
    for word in re.findall(r"[A-Za-z]{4,12}", text):
        if _levenshtein(word.lower(), target) <= max_distance:
            return True
    return False


def looks_like_fee(text: str) -> bool:
    """Cheap hint that OCR text is probably a fee receipt page."""
    return bool(text) and (_fuzzy_word_present(text, "receipt")
                           or _fuzzy_word_present(text, "waiver"))


def has_fee_value(text: str) -> bool:
    """True iff parsing recovers a fee_status evidence value."""
    return "fee_status" in parse_page(text or "", "ocr").fields


def looks_like_biometric(text: str) -> bool:
    """Cheap hint that OCR text is probably a biometric scan slip."""
    return bool(text) and (_fuzzy_word_present(text, "biometric")
                           or "b-13" in text.lower())


def has_observed_flags_line(text: str) -> bool:
    """True iff parsing recovers an observed_flags evidence value."""
    return "observed_flags" in parse_page(text or "", "ocr").fields


def _parse_finding(flat: str, add) -> None:
    flat = _canonicalize_note_labels(flat)
    m = _FINDING_RE.search(flat)
    if not m:
        return
    verdict = _normalize_verdict(m.group(1))
    if verdict:
        add("note_verdict", verdict, long_ok=False)
    reason = m.group(2).strip()
    if reason:
        add("note_reason", reason, long_ok=True)


# Sponsor attestation body, e.g.:
#   "Sponsor SPN-4974 attests that Aririx Arizarn is expected on Earth for
#   cultural exchange. The sponsor acknowledges responsibility for class
#   XW-1 compliance and immediate reporting duties."
_ATTEST_SPONSOR_NAME_RE = re.compile(
    r"(?i)Sponsor\s+(\S+)\s+attests that\s+(.+?)\s+is expected on Earth for"
)
_ATTEST_VISA_RE = re.compile(r"(?i)\bclass\s+(.+?)\s+compliance")


def _parse_attestation(flat: str, add) -> None:
    m = _ATTEST_SPONSOR_NAME_RE.search(flat)
    if m:
        add("attest_sponsor", m.group(1), long_ok=False)
        add("attest_name", m.group(2), long_ok=False)
    m = _ATTEST_VISA_RE.search(flat)
    if m:
        add("attest_visa", m.group(1), long_ok=False)


# Manual correction lines, e.g. "Manual correction: sponsor is SPN-2191."
_CORRECTION_RE = re.compile(
    r"(?i)Manual correction:\s*([a-zA-Z ]+?)\s+is\s+(.+?)\.(?:\s|$)"
)

_CORRECTION_WHAT_MAP = {
    "sponsor": "sponsor_id",
    "visa class": "visa_class",
    "species": "species_code",
    "home world": "home_world",
    "arrival date": "arrival_date",
    "applicant": "applicant_name",
    "purpose": "declared_purpose",
    "declared purpose": "declared_purpose",
    "fee": "fee_status",
    "fee status": "fee_status",
}


def _parse_corrections(flat: str) -> dict:
    corrections: dict[str, str] = {}
    for m in _CORRECTION_RE.finditer(flat):
        what = m.group(1).strip().lower()
        field_name = _CORRECTION_WHAT_MAP.get(what)
        if field_name is None:
            # Unrecognized "<what>" -- ignore rather than guess.
            continue
        value = _clean_value(m.group(2), long_ok=False)
        if value is None:
            continue
        corrections[field_name] = value
    return corrections


# --- entry point -----------------------------------------------------------


def parse_page(text: str, source: str) -> PageParse:
    """Parse one page's text into a doc type, field evidence, and any
    manual corrections found on the page.

    `source` is caller-supplied provenance ("text" or "ocr") and is stored
    verbatim in every Evidence tuple produced from this call; parse_page
    does not itself decide which text bucket to prefer.
    """
    text = text or ""
    lines = text.split("\n")
    doc_type = _detect_doc_type(lines)
    fields: dict[str, list[Evidence]] = {}

    def add(field_name: str, raw: str, *, long_ok: bool = False) -> None:
        cleaned = _clean_value(raw, long_ok=long_ok)
        if cleaned is None:
            return
        fields.setdefault(field_name, []).append((cleaned, doc_type, source))

    _parse_labeled_fields(lines, add)

    flat = _flatten(text)
    _parse_finding(flat, add)
    _parse_attestation(flat, add)
    corrections = _parse_corrections(flat)

    return PageParse(doc_type=doc_type, fields=fields, corrections=corrections)
