# MIB Doc Challenge — Solution

Offline document-processing pipeline for the [8090 MIB Doc Challenge](https://github.com/8090-inc/mib-doc-challenge):
given a directory of PDF case packets, it extracts each applicant's record,
adjudicates the case (`APPROVED` / `DENIED` / `NEEDS_REVIEW`), and writes one
JSONL row per case with a calibrated confidence. Everything runs offline in
Docker on CPU — no LLMs, no network, no cloud OCR.

## Architecture

Each PDF flows through five stages: **extract** pulls per-span text with a
visibility filter that quarantines hidden text (white-on-white, alpha-0 /
invisible render mode, sub-3pt, outside the cropbox — the dataset's
prompt-injection vectors) and OCRs image-only pages with a measured two-pass
Tesseract ladder plus a domain word list; **parse** classifies each page by
form type and reads labeled fields with OCR-noise-tolerant label matching;
**resolve** merges evidence across pages by the field manual's precedence
(manual corrections > adjudicator note > intake form > biometric slip >
attestation > registry), snaps every value to closed vocabularies with
tie-refusal, discards foreign-applicant pages, and keeps unknowns as unknowns;
**rules** maps the resolved record through a policy cascade (deterministic
rules mined from the public training labels, then learned class distributions
keyed by rule path and evidence quality); **decide** picks the adjudication
that maximizes expected evaluator points and emits a confidence equal to the
out-of-fold accuracy of that rule path, with a guard that never approves when
the learned denial probability exceeds 15%.

| Module | Responsibility |
| --- | --- |
| `mib/extract.py` | Visible/hidden span split, image-only detection, OCR ladder |
| `mib/parse.py` | Per-page doc-type detection and field parsing |
| `mib/vocab.py` | Closed vocabularies + edit-distance snapping |
| `mib/resolve.py` | Cross-page precedence merge, conflicts, evidence summary |
| `mib/rules.py` | Adjudication cascade (deterministic + learned paths) |
| `mib/decide.py` | Expected-points decision + calibrated confidence |
| `mib/output.py` | Validator-safe row building, JSONL writer |
| `mib/main.py` | CLI, 4-way multiprocessing, per-PDF fallback |

## Run with Docker

```bash
docker build -t mib-submission .
mkdir -p /tmp/mib-output
docker run --rm --network none \
  --mount type=bind,src=/path/to/pdfs,dst=/input,readonly \
  --mount type=bind,src=/tmp/mib-output,dst=/output \
  mib-submission /input /output/predictions.jsonl
```

The image accepts exactly two arguments (`<input_pdf_dir> <output_predictions_path>`),
needs no network, runs on 4 vCPU / 8 GiB, writes only to `/tmp` and the output
mount, and produces byte-identical output across runs.

## Learned artifacts and provenance

All checked-in artifacts are fitted on the **public training labels only** and
regenerate deterministically — nothing is keyed to case IDs or file names:

```bash
python3 tools/build_vocab.py       # mib/data/vocab.json from train_labels.csv
python3 tools/build_user_words.py  # mib/data/user_words.txt from vocab.json
python3 tools/fit_stats.py --records <resolved.jsonl>  # mib/data/policy_stats.json
python3 tools/mine_policy.py       # audit: re-derives every constant in mib/rules.py
```

`tools/run_train.py`, `tools/make_predictions.py`, and
`tools/score_extraction.py` reproduce the full local evaluation loop against
the challenge repo's `scripts/evaluate.py`.

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

148 tests cover the injection defenses (including synthetic invisible-render-mode
attacks), parsing fixtures from real packets, precedence/conflict rules, the
decision math, and end-to-end hardening (corrupt PDFs, sorted deterministic
output).

## License

MIT — see [LICENSE](LICENSE).
