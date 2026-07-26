#!/usr/bin/env python3
"""Regenerate mib/data/user_words.txt for Tesseract's --user-words.

Provenance: the word list is derived entirely from mib/data/vocab.json (itself
regenerated from the public training labels by tools/build_vocab.py) plus a
fixed set of generic form-label words that appear on the synthetic MIB form
templates ("Case", "Sponsor", "Finding", ...). It contains vocabulary tokens
only — no case IDs, no answers, nothing keyed to specific PDFs — and biases
OCR toward the document domain for any fresh packet.

Usage: python3 tools/build_user_words.py  (writes mib/data/user_words.txt)
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "mib" / "data" / "vocab.json"
OUT_PATH = ROOT / "mib" / "data" / "user_words.txt"

# Generic wording of the six MIB form templates (labels/headers), not learned
# from answers. Kept explicit so the derivation is reviewable.
FORM_LABEL_WORDS = [
    "Adjudicator", "Applicant", "Arrival", "Attestation", "Biometric",
    "Case", "Class", "Code", "Date", "Declared", "Extract", "Fee", "Finding",
    "Home", "Intake", "MIB", "Match", "Note", "Observed", "Purpose", "Reason",
    "Receipt", "Registry", "SPN", "Slip", "Species", "Sponsor", "Status",
    "Visa", "Waiver", "World", "confidence", "flags",
]


def build_words() -> list:
    vocab = json.loads(VOCAB_PATH.read_text())
    tokens = set(FORM_LABEL_WORDS)
    for values in vocab.values():
        for value in values:
            for token in re.split(r"[^0-9A-Za-z]+", value):
                if token:
                    tokens.add(token)
    return sorted(tokens)


def main():
    words = build_words()
    OUT_PATH.write_text("\n".join(words) + "\n")
    print(f"wrote {OUT_PATH} ({len(words)} words)")


if __name__ == "__main__":
    main()
