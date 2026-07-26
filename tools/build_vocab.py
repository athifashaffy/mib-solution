"""Regenerate mib/data/vocab.json from the public training labels CSV.

Provenance: the closed vocabularies snapped against by mib.vocab are learned
once, offline, from the challenge repo's train_labels.csv (columns:
case_id, applicant_name, species_code, home_world, visa_class, sponsor_id,
arrival_date, declared_purpose, risk_flags, fee_status, adjudication). This
is a one-time, human-run regeneration step, not something the pipeline does
at inference time: the resulting mib/data/vocab.json is checked in and
loaded read-only by mib.vocab. The vocabularies describe the closed universe
of valid field values that can appear on ANY input packet -- nothing here is
keyed to, or looked up by, individual case IDs at runtime.

Usage:
    python -m tools.build_vocab [--labels PATH] [--out PATH]

Defaults to ../mib-doc-challenge/data/train_labels.csv relative to this
repo's root (i.e. a sibling checkout of the challenge repo) and writes to
mib/data/vocab.json.
"""
import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS_PATH = REPO_ROOT.parent / "mib-doc-challenge" / "data" / "train_labels.csv"
DEFAULT_OUT_PATH = REPO_ROOT / "mib" / "data" / "vocab.json"

# risk_flags is a "|"-delimited multi-value column; "none" is a sentinel for
# "no flags on this case", not a vocabulary entry.
NO_FLAG_SENTINEL = "none"


def build_vocab(labels_path) -> dict:
    """Read train_labels.csv at labels_path and derive the closed
    vocabularies as sorted, deduplicated lists of canonical strings."""
    species: set[str] = set()
    world: set[str] = set()
    purpose: set[str] = set()
    visa: set[str] = set()
    fee: set[str] = set()
    flag: set[str] = set()
    first_names: set[str] = set()
    last_names: set[str] = set()

    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            species.add(row["species_code"])
            world.add(row["home_world"])
            purpose.add(row["declared_purpose"])
            visa.add(row["visa_class"])
            fee.add(row["fee_status"])
            for entry in row["risk_flags"].split("|"):
                entry = entry.strip()
                if entry and entry != NO_FLAG_SENTINEL:
                    flag.add(entry)

            name_tokens = row["applicant_name"].split()
            if len(name_tokens) == 2:
                first_names.add(name_tokens[0])
                last_names.add(name_tokens[1])

    return {
        "species": sorted(species),
        "world": sorted(world),
        "purpose": sorted(purpose),
        "visa": sorted(visa),
        "fee": sorted(fee),
        "flag": sorted(flag),
        "first_names": sorted(first_names),
        "last_names": sorted(last_names),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS_PATH,
        type=Path,
        help="Path to train_labels.csv (default: sibling mib-doc-challenge checkout)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_PATH,
        type=Path,
        help="Path to write vocab.json (default: mib/data/vocab.json)",
    )
    args = parser.parse_args(argv)

    vocab = build_vocab(args.labels)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {args.out} from {args.labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
