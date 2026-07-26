# MIB Solution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offline Dockerized pipeline that reads PDF case packets and emits scored predictions.jsonl, targeting ≥130/150 on the public train set.

**Architecture:** PyMuPDF visible-text extraction with span-level visibility filtering (injection defense) → Tesseract OCR for image-only pages → per-doc-type field parsing with provenance → precedence-based cross-page resolution with closed-vocab snapping → mined rule cascade emitting class probabilities → expected-points decision with empirically calibrated confidence.

**Tech Stack:** Python 3.12, PyMuPDF, Tesseract 5 (subprocess), pytest. No network, no ML frameworks, no model artifacts.

## Global Constraints

- Container contract: `docker run --rm --network none <image> /input /output/predictions.jsonl`; read-only rootfs, writable `/tmp` and `/output` only.
- Runtime: 4 vCPU / 8 GiB; ≤6 s/PDF average (target ≤2 s/PDF).
- Every row must pass `validate_submission.py`: all 12 fields present; `sponsor_id` matches `SPN-\d{4}` (unknown → `SPN-0000`); `arrival_date` valid ISO (unknown → `1900-01-01`); `fee_status` ∈ {paid,waived,unpaid,unknown}; adjudication ∈ {APPROVED,DENIED,NEEDS_REVIEW}; confidence ∈ [0,1] JSON number; only case IDs from input PDFs.
- Never omit a case; per-PDF exceptions degrade to NEEDS_REVIEW fallback row.
- Deterministic: sorted output, pinned deps, no wall-clock/randomness in the decision path.
- Challenge repo (`../mib-doc-challenge`) is read-only reference; all code lives in `mib-solution/`.
- Every commit message ends with the standard Co-Authored-By/Claude-Session trailer.

---

### Task 1: Scaffold + end-to-end skeleton

**Files:**
- Create: `requirements.txt`, `Dockerfile`, `run.sh`, `mib/__init__.py`, `mib/output.py`, `mib/main.py`, `tests/test_output.py`, `.gitignore`, `pytest.ini`

**Interfaces:**
- Produces: `mib.output.build_row(case_id, fields: dict, adjudication: str, confidence: float) -> dict` (fills validator-safe placeholders for missing keys); `mib.output.write_jsonl(rows, path)`; `mib.main.process_dir(input_dir, output_path)`; CLI `python -m mib.main <input_dir> <output_path>`.
- `FALLBACK_ROW` semantics: unknown fields → `applicant_name/species_code/home_world/visa_class/declared_purpose = "unknown"`, `sponsor_id="SPN-0000"`, `arrival_date="1900-01-01"`, `risk_flags="none"`, `fee_status="unknown"`.

- [ ] Write `tests/test_output.py`: `build_row("MIB-000001", {}, "NEEDS_REVIEW", 0.3)` produces a dict passing a local copy of the validator row checks; unknown placeholders as above; run to see it fail.
- [ ] Implement `mib/output.py` + `mib/main.py` (serial loop over `sorted(Path(input).glob("*.pdf"))`, all rows fallback NEEDS_REVIEW).
- [ ] `pytest -q` passes.
- [ ] Write `Dockerfile` (python:3.12-slim, `apt-get install tesseract-ocr`, pip install pinned reqs, copy `mib/`, `ENTRYPOINT ["/app/run.sh"]`) and `run.sh`.
- [ ] Verify: `docker build`, run on 5 copied train PDFs with `--network none --read-only --tmpfs /tmp`, then challenge `validate_submission.py --pdf-dir` passes.
- [ ] Commit.

### Task 2: extract.py — spans, visibility, OCR

**Files:**
- Create: `mib/extract.py`, `tests/test_extract.py`

**Interfaces:**
- Produces:
  - `extract_document(pdf_path) -> DocText` where `DocText.pages: list[PageText]`,
    `PageText.visible: str` (text-layer visible text), `PageText.hidden: str`,
    `PageText.ocr: str | None` (filled only for image-only pages),
    `PageText.image_only: bool`.
  - Visibility rule: a span is hidden iff luminance(color) > 240, or size < 3,
    or bbox fully outside the page cropbox.
  - `PageText.image_only` iff visible chars < 100 and page has images.
  - OCR: render page at 200 dpi grayscale via PyMuPDF `get_pixmap`, write PNG to
    `/tmp`, run `tesseract png stdout --psm 6 -l eng` via subprocess with 20 s
    timeout; on nonzero exit or timeout, `ocr=""`. If OCR yields < 50 chars,
    retry with `--psm 3`.
- Consumes: nothing from other tasks.

- [ ] Tests (real data paths, skip-if-missing): MIB-000042 page 0 hidden text contains "SYSTEM:", visible does not; MIB-000395 page 1 is image_only and OCR text contains "I-8090" and "Tektari".
- [ ] Implement; `pytest -q`; commit.

### Task 3: vocab.py — closed vocabularies + snapping

**Files:**
- Create: `mib/vocab.py`, `tests/test_vocab.py`, `tools/build_vocab.py`, `mib/data/vocab.json`

**Interfaces:**
- Produces: `snap(value, kind) -> str | None` for kinds `species|world|visa|purpose|fee|flag`; `snap_name(value) -> str | None` (token-wise against first/last vocab); `VOCAB` loaded from packaged `mib/data/vocab.json`. Snapping = case-insensitive exact else min edit distance with threshold ≤ 2 (≤ 3 for len ≥ 10), ties → None.
- `tools/build_vocab.py` regenerates `vocab.json` from `../mib-doc-challenge/data/train_labels.csv` (checked-in artifact; tool documents provenance).

- [ ] Tests: `snap("JOVIAN_GAS M","species")=="JOVIAN_GASFORM"`; `snap("TRANS1T-7","visa")=="TRANSIT-7"`; `snap("Prox1ma-b","world")=="Proxima-b"`; `snap_name("Tektari So1ix")=="Tektari Solix"`; garbage → None.
- [ ] Implement + generate artifact; `pytest -q`; commit.

### Task 4: parse.py — doc types, fields, corrections, notes

**Files:**
- Create: `mib/parse.py`, `tests/test_parse.py`

**Interfaces:**
- Produces: `parse_page(text: str, source: str) -> PageParse` with:
  - `doc_type` ∈ {intake, fee, biometric, sponsor, registry, note, cover, unknown} via header keywords ("FORM I-8090", "MIB Fee Receipt", "Biometric Scan Slip", "Sponsor Attestation", "Planetary Registry Extract", "Manual Adjudicator Note").
  - `fields: dict[str, list[Evidence]]`, `Evidence = (value_raw, doc_type, source)` for the 8 non-flag fields plus `fee_status`, `waiver_code`, `observed_flags`, `registry_status`, `note_verdict`, `note_reason`, `attest_name`, `attest_sponsor`, `attest_visa`.
  - `corrections: dict[field, value]` from "Manual correction: <what> is <value>."
  - Handles both text-layer layout (label and value separated by newlines/pipes) and OCR layout ("Label: value" with OCR noise like `0/O`, `1/l`).
- Consumes: nothing (pure text in).

- [ ] Tests built from real extracted strings (copy literals into test file): intake form parse of MIB-000912 page 0 text; correction parse of MIB-000777 ("sponsor is SPN-2191"); note verdict parse of MIB-000431 ("Finding: DENIED"); OCR-noise variant "Sp0nsor ID: SPN-8l46" → raw captured (snapping is resolve's job).
- [ ] Implement; `pytest -q`; commit.

### Task 5: resolve.py — precedence merge

**Files:**
- Create: `mib/resolve.py`, `tests/test_resolve.py`

**Interfaces:**
- Consumes: `DocText` (Task 2), `parse_page` (Task 4), `vocab.snap` (Task 3).
- Produces: `resolve_case(doc: DocText) -> CaseRecord` with `fields: dict[str,str|None]` (snapped, trusted-only), `evidence: EvidenceSummary` (`note_verdict`, `note_reason`, `waiver_code`, `observed_flags: set`, `registry_status`, `has_image_only_pages`, `conflicts: set`, `ocr_used`, `hidden_only_fields: set`).
- Precedence per field: manual correction > note > intake > biometric > sponsor-attestation > registry; text-layer beats OCR at equal doc rank; hidden text never supplies a value (field with only-hidden evidence → None + membership in `hidden_only_fields`).
- Conflict detection: same-rank trusted disagreement after snapping → add `identity_conflict` (names) / `sponsor_mismatch` (sponsor) to `conflicts`.

- [ ] Tests: hidden-only value resolves None; correction beats form; conflicting sponsor across intake vs attestation yields `sponsor_mismatch`; OCR value used when field absent from text layer.
- [ ] Implement; `pytest -q`; commit.

### Task 6: extraction scoring loop (milestone: weighted extraction ≥ 90% on train)

**Files:**
- Create: `tools/score_extraction.py`, `tools/run_train.py`

**Interfaces:**
- `tools/run_train.py --limit N --jobs 4 --out /tmp/mib-train.jsonl` runs the real pipeline (fields only; adjudication placeholder NEEDS_REVIEW/0.3) over train PDFs.
- `tools/score_extraction.py` compares against `train_labels.csv` with the evaluator's normalize rules; prints per-field accuracy and top-20 mismatches with case ids.

- [ ] Build tools; run on 100 PDFs; fix parse/snap bugs revealed; then full 1,000.
- [ ] Iterate until weighted extraction (evaluator weights) ≥ 90%; each fix gets a test if it's a code rule, and a commit.

### Task 7: rules.py + decide.py + calibration

**Files:**
- Create: `mib/rules.py`, `mib/decide.py`, `tools/fit_stats.py`, `mib/data/policy_stats.json`, `tests/test_rules.py`, `tests/test_decide.py`

**Interfaces:**
- `rules.adjudicate(record: CaseRecord) -> RulePath` where `RulePath.path_id: str` and `RulePath.dist: (pA, pD, pR)`. Cascade (first match wins):
  1. note_verdict present → dist (1,0,0)/(0,1,0)/(0,0,1) per verdict, path `note:<verdict>`
  2. risk flags ∩ disqualifying → DENIED
  3. visa TRANSIT-7 → DENIED
  4. fee unpaid → DENIED; fee unknown/None → NEEDS_REVIEW
  5. sponsor in revoked list and visa ≠ DIP-1 → DENIED
  6. sponsor missing and visa ≠ DIP-1 → NEEDS_REVIEW
  7. arrival_date missing or hidden-only → NEEDS_REVIEW
  8. remaining paths keyed by (flag-set profile, fee_status, waiver_code, visa
     group, stale-date bucket, conflicts, illegibility signals) → empirical dist
     from `policy_stats.json`
- `tools/fit_stats.py`: runs pipeline over train, groups truth adjudication by `path_id`, writes Laplace-smoothed distributions + per-path decision accuracy via 5-fold CV → `policy_stats.json` (checked in, provenance documented).
- `decide.decide(rule_path) -> (adjudication, confidence)`: expected points `E[A]=8pA−4pD+1pR`, `E[D]=8pD+1pR`, `E[R]=8pR+2(pA+pD)`; argmax; confidence = CV accuracy of that decision on that path (fallback: probability of chosen class), clamped to [0.05, 0.99].

- [ ] Unit tests for cascade order and decision math with hand-computed expectations (e.g. dist (0.5,0.3,0.2): E[A]=2.6, E[D]=2.6, E[R]=3.2 → NEEDS_REVIEW).
- [ ] Fit stats, wire into main, full train run + `evaluate.py`. Milestone: classification ≥ 68/80, calibration ≥ 15/20, 0 catastrophic false approvals target (≤2 acceptable).
- [ ] Commit (artifacts included).

### Task 8: hardening — parallelism, fallback, determinism

**Files:**
- Modify: `mib/main.py`; Create: `tests/test_main.py`

- [ ] `multiprocessing.Pool(min(4, cpu_count))`, per-PDF try/except → fallback row, imap_unordered then sort by case_id; corrupted-PDF test (truncated file → fallback row present).
- [ ] Two consecutive full-train runs byte-identical (`cmp`).
- [ ] Commit.

### Task 9: residual squeeze loop (milestone: total ≥ 130)

**Files:**
- Create: `tools/residuals.py` (dumps wrong-adjudication and missed-field cases grouped by cause with page text snippets)

- [ ] Iterate: run train → evaluate → inspect top residual clusters → fix (parse gaps, new rule-path features like waiver codes/staleness reference, OCR retry knobs) → refit stats → rerun. Stop when total ≥ 130 or marginal gain < 0.5/iteration.
- [ ] Guard each substantive fix with a unit test; commit per iteration.

### Task 10: Docker validation run + submission packaging

**Files:**
- Create: `MEMO.md` (repo root), `submission_out/` (gitignored)

- [ ] Full Docker run on `data/train` under exact scoring flags via `run_docker_submission.py`; confirm score parity with local run and ≤6 s/PDF.
- [ ] Full Docker run on `data/validation` (5,000 PDFs); `validate_submission.py --manifest` passes.
- [ ] Write `MEMO.md` (approach, evidence-mined policy artifacts + provenance, failure modes, next-week improvements).
- [ ] Stage `submissions/<username>/{predictions.jsonl, MEMO.md, SUBMISSION.md}` content in `submission_out/` for the user to PR (username pending).
- [ ] Final commit.
