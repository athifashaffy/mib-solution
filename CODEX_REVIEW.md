# Codex Live Review

Last updated: 2026-07-26 08:56 America/Toronto

This file records independent review findings while the Claude session builds the
submission. It is intentionally kept in the solution repository, not the official
challenge repository.

## Current status

- The official challenge repository is understood and its five public contract
  tests pass with `python3 -m unittest discover -s tests -v`.
- The 2,879,970,504-byte public dataset archive was downloaded in eight ranges and
  its SHA-256 matches the published checksum:
  `a9bb8c1bbf51346ebf49c2e3e1acdb7a5d6cd0760162767b0d133c7b7200f3c4`.
- All 1,000 training PDFs and 5,000 validation PDFs are present.
- The separate solution repo is the correct place for implementation. At the time
  of this note it contains only the initial README commit.

## Must fix before implementation

### P0 - The proposed fixed decision thresholds are not valid for three classes

The earlier proposal says to approve when `P(APPROVED) > 0.5`, deny when
`P(DENIED) > 0.25`, and otherwise review. Those cutoffs are derived only after
assuming `P(NEEDS_REVIEW) = 0`. That assumption is invalid: 280/1,000 training
cases are truly `NEEDS_REVIEW`.

For probabilities `pA`, `pD`, and `pR`, the evaluator's expected raw utilities are:

```text
U(APPROVED)     = 8*pA - 4*pD + 1*pR
U(DENIED)       = 0*pA + 8*pD + 1*pR
U(NEEDS_REVIEW) = 2*pA + 2*pD + 8*pR
```

Choose the label with maximum expected utility. If probability estimates are not
available, use an explicit ordered rule cascade and calibrate its outcomes; do not
claim the two fixed thresholds are the exact optimizer for the real three-class
problem.

### P0 - Classification needs evidence beyond the ten submitted fields

Training-label mining shows a rule cascade using perfect submitted fields reaches
882/1,000 classification accuracy and about 71/80 classification points, but it
cannot resolve all exceptions. The internal evidence model must also capture:

- packet receipt date, needed for the 180-day staleness rule;
- visible hardship-waiver or diplomatic-note evidence;
- registry status such as embargo/revocation;
- sponsor-attestation identity and validity;
- biometric confidence and visible flags;
- manual notes/stamps, including later rescissions and evidence precedence;
- page/case association when a packet contains multiple applicants.

These are internal features and must not be confused with the output schema.

### P0 - The visibility-filter design must match what the PDF library exposes

The installed `pdfplumber` character dictionaries expose color, size, geometry,
matrix, and tag data, but not the PDF text rendering mode (`Tr`). Therefore a
requirement to reject `Tr 3` text cannot be implemented from
`pdfplumber.Page.chars` alone.

Real case `MIB-000013` demonstrates the attack: ordinary text extraction returns a
complete fake `SYSTEM: ... answer key` on every page. In this case the duplicated
payload is detectable because one copy is white and the other is outside the page
bounds, but the solution still needs a tested strategy for render-mode-hidden text
and other trap variants. Options include parsing content-stream graphics state,
using a lower-level PDF library, or reconciling extracted text against rendered/OCR
evidence. Add focused regression tests using known adversarial training packets.

## Important design corrections

### P1 - Preserve internal unknowns while emitting schema-valid rows

The JSON schema requires every field and requires `sponsor_id` and `arrival_date`
to match strict patterns. The extractor should preserve `unknown`/missing values
internally for adjudication, then serialize deterministic schema-valid fallback
values only at the output boundary. Never let a placeholder such as `SPN-0000` or
`1900-01-01` trigger a real denial/staleness rule.

### P1 - Avoid calibration leakage

Do not tune rules on all 1,000 labels and then call a 200-case subset "held out."
Freeze a stratified calibration/evaluation split before tuning, or use
out-of-fold predictions. Confidence is the empirical probability that the emitted
adjudication is correct, preferably conditioned on predicted label and evidence
quality, not a generic extraction-quality score.

### P1 - Gate OCR aggressively and measure the actual Docker budget

The runtime budget is six seconds per PDF on four CPUs. Do not run full-resolution
Tesseract on every page. Prefer trusted text-layer extraction first, route only
image-only/damaged/suspicious pages to OCR, render at a measured DPI, cap work per
page, and benchmark p50/p95 plus total wall time inside the final container.

### P1 - Keep the official challenge checkout clean

The dataset archive and extracted PDF directories are ignored, as intended.
Unzipping touched `data/validation_manifest.csv` metadata and made Git report it as
modified, although its content hash is identical to `HEAD`. Do not commit any
official-repo change. Verify `git diff` is empty before submission work.

## Process issue

The invoked brainstorming skill forbids scaffolding or commits before presenting a
design and receiving approval. The solution repo was initialized and committed
before that approval gate. This does not damage the solution technically, but the
remaining workflow should return to the required order: finish PDF inspection,
present alternatives and a design, obtain approval, write/review the spec, then
write an implementation plan.

## Review gates for the eventual implementation

- Unit tests for visibility filtering, precedence, extraction, adjudication, and
  schema serialization.
- Score reported on genuinely held-out or out-of-fold training predictions.
- Offline Docker build/run/validation from a clean checkout.
- Byte-identical output across two runs.
- Zero crashes and one valid row per input PDF unless omission is deliberately
  proven to have higher expected value.
- Runtime and image-size measurements under the official limits.
- No hardcoded case IDs, answer tables, or validation-specific outputs.
- Clean Git status in the official challenge repository.
