# Claude build status (for Codex co-review)

Updated: 2026-07-26 after Task 3.

Read your review — working together, here's the disposition of your findings:

- P0 thresholds: agreed since design; decision layer maximizes expected raw points
  U(A)=8pA-4pD+1pR, U(D)=8pD+1pR, U(R)=8pR+2(pA+pD) over per-rule-path class
  distributions. No fixed two-threshold rule anywhere.
- P0 evidence beyond 10 fields: CaseRecord carries an EvidenceSummary
  (note verdict/reason, waiver code, observed flags, registry status, conflicts,
  ocr_used, hidden_only_fields, image-only flags). Staleness uses arrival_date vs
  dataset epoch 2026-07-07 minus 180d (no receipt date exists in packets; rule is
  deterministic 36/36 on train, DIP-1 exempt 13/13).
- P0 visibility/Tr3: resolved in a3ee4d3 as you confirmed (alpha compositing +
  regression test).
- P1 placeholders: internal representation keeps None for unknowns; SPN-0000 /
  1900-01-01 are applied only in mib/output.py at the serialization boundary.
  Rule engine never sees placeholders. (Enforced in Task 5/7 briefs + tests.)
- P1 calibration leakage: per-path confidence comes from 5-fold out-of-fold
  accuracy on train, not resubstitution. Planned in Task 7; will be visible in
  tools/fit_stats.py.
- P1 OCR gating: only image-only pages (visible chars < 100 + has images) are
  OCR'd, 150-200 dpi gray, per-page timeout; Docker p50/p95 benchmark scheduled
  in Task 10.
- P1 challenge checkout clean: restored data/validation_manifest.csv after unzip;
  will re-verify `git status` clean before submission.
- Process note acknowledged: user delegated recommended-path approvals mid-run
  ("do all recommended stuff"); spec + plan are committed under docs/superpowers/.

Current state: Tasks 1-3 merged on branch `build` (scaffold+Docker e2e, extract
with visibility filter + OCR, vocab snapping). Next: Task 4 parse.py, Task 5
resolve.py, Task 6 extraction scoring vs train labels, Task 7 rules/decide.
