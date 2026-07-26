"""Closed vocabularies + edit-distance snapping for MIB submission fields.

`VOCAB` is loaded once at import time from the packaged artifact
`mib/data/vocab.json`. That artifact is learned offline from the public
challenge training labels (mib-doc-challenge/data/train_labels.csv) via
`tools/build_vocab.py`; see that module's docstring for full provenance.
The vocabularies describe the closed universe of valid values a field can
take on ANY input packet -- they are not keyed to, or looked up by,
individual case IDs at runtime. Regenerating vocab.json is a one-time,
human-run step, not something this module does.

Snapping semantics (case-insensitive):
  1. Exact match (case-insensitively) against a vocabulary entry returns
     that entry's canonical form.
  2. Otherwise, compute Levenshtein distance (case-insensitively) between
     the cleaned input and every vocabulary entry. The closest entry is
     accepted if its distance is <= 2 (or <= 3 when the cleaned input is
     at least 10 characters long).
  3. If two or more entries tie for the minimum accepted distance, the
     match is ambiguous and snap() returns None.
  4. Input is whitespace-collapsed and stripped before matching; empty or
     None input always returns None.
"""
import json
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent / "data" / "vocab.json"

with open(_DATA_PATH) as _f:
    VOCAB: dict = json.load(_f)

# Below this length, a distance <= 2 is accepted; at/above it, <= 3 is
# accepted. Longer strings have more room for OCR noise to accumulate
# without the candidate becoming genuinely ambiguous.
_SHORT_THRESHOLD = 2
_LONG_THRESHOLD = 3
_LONG_MIN_LENGTH = 10


def _levenshtein(a: str, b: str) -> int:
    """Standard dynamic-programming edit distance (insert/delete/substitute,
    unit cost) between two strings. No stdlib/third-party dependency."""
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
                prev_row[j] + 1,  # deletion
                cur_row[j - 1] + 1,  # insertion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = cur_row
    return prev_row[-1]


def _threshold_for(cleaned: str) -> int:
    return _LONG_THRESHOLD if len(cleaned) >= _LONG_MIN_LENGTH else _SHORT_THRESHOLD


def _clean(value) -> str | None:
    """Whitespace-collapse and strip; None/empty (post-clean) -> None."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def snap(value, kind: str) -> str | None:
    """Snap value against the closed vocabulary for `kind`. Returns the
    canonical vocab entry, or None if no acceptable/unambiguous match."""
    cleaned = _clean(value)
    if cleaned is None:
        return None

    entries = VOCAB.get(kind)
    if not entries:
        return None

    lowered = cleaned.lower()

    for entry in entries:
        if entry.lower() == lowered:
            return entry

    threshold = _threshold_for(cleaned)
    best_distance = None
    best_entries: list[str] = []
    for entry in entries:
        distance = _levenshtein(lowered, entry.lower())
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_entries = [entry]
        elif distance == best_distance:
            best_entries.append(entry)

    if best_distance is None or best_distance > threshold:
        return None
    if len(best_entries) != 1:
        # Ambiguous: two+ vocab entries tie at the minimum distance.
        return None
    return best_entries[0]


def snap_name(value) -> str | None:
    """Snap a two-token "First Last" applicant name against the
    first_names/last_names vocabularies independently, token-wise. Returns
    "First Last" in canonical form, or None if the input isn't exactly two
    tokens or either token fails to snap."""
    cleaned = _clean(value)
    if cleaned is None:
        return None

    tokens = cleaned.split(" ")
    if len(tokens) != 2:
        return None

    first = snap(tokens[0], "first_names")
    last = snap(tokens[1], "last_names")
    if first is None or last is None:
        return None
    return f"{first} {last}"
