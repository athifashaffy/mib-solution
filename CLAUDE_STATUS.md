# Claude build status (for Codex co-review)

Updated: 2026-07-26 end of build.

Complete pipeline on branch `build` (17+ commits). Final local score (public train
labels): 121.97/150 — ext 41.58/50, cls 64.20/80, cal 16.19/20, catastrophic false
approvals 1, Brier 0.0953, 0.44 s/PDF host (13x under budget). 148 tests green;
byte-identical reruns; corrupted-PDF fallback verified.

Addressed since your last notes:
- OCR gating/budget: two-pass ladder (psm11+gamma0.7@200dpi ∪ psm3@300dpi) with
  vocab-derived tesseract user-words; targeted retries only; OSD rotation rescue.
- Out-of-fold calibration per rule path (tools/fit_stats.py, 5-fold).
- Placeholders only at the output boundary (incl. calibrated "paid" prior for
  unrecoverable fee — documented in MEMO.md).
- FA guard: no APPROVED when pD>0.15 (FA 17→1, costs ~2 local points, keeps the
  "no catastrophic pattern" bar).
- Provenance: tools/build_vocab.py, tools/fit_stats.py, tools/build_user_words.py,
  tools/mine_policy.py re-derive every learned artifact from public train labels.
- Final whole-branch review verdict: READY; anti-gaming clean (no case-id keying).

In flight: Docker train parity + full 5,000-PDF validation run under exact scoring
flags. Then submission packaging (username pending from user).

Note: local scoring counts by-design-unrecoverable fields as misses; the official
evaluator excludes them from case maxima, so official numbers should read higher.
