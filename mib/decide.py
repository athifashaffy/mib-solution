"""Expected-points decision layer.

The evaluator's raw classification payoffs are:
  correct = 8; NEEDS_REVIEW on a truly A/D case = 2; A/D on a truly
  NEEDS_REVIEW case = 1; wrong A-vs-D = 0; APPROVED on a truly DENIED
  case = -4.
So for class probabilities (pA, pD, pR):
  E[APPROVED]     = 8*pA - 4*pD + 1*pR
  E[DENIED]       = 8*pD + 1*pR
  E[NEEDS_REVIEW] = 8*pR + 2*(pA + pD)
We emit the argmax. Confidence is the out-of-fold accuracy of that path's
decision measured on the public train labels (proper for the Brier-scored
calibration section), falling back to the chosen class's probability when a
path has no fitted stats.
"""
import json
from pathlib import Path

CONF_MIN = 0.05
CONF_MAX = 0.99

# Catastrophic-false-approval guard: even when expected points favor
# APPROVED, refuse to approve while the path's denial probability exceeds
# this ceiling and hedge to NEEDS_REVIEW instead. The challenge tie-breakers
# rank catastrophic false-approval count explicitly and the interview bar
# requires "no catastrophic false-approval pattern", both of which price
# false approvals beyond the -4 raw points the expected-value math sees.
FA_GUARD_MAX_PD = 0.15

_LABELS = ("APPROVED", "DENIED", "NEEDS_REVIEW")

_STATS_PATH = Path(__file__).resolve().parent / "data" / "policy_stats.json"
_stats_cache = None


def load_stats() -> dict:
    global _stats_cache
    if _stats_cache is None:
        if _STATS_PATH.exists():
            _stats_cache = json.loads(_STATS_PATH.read_text())
        else:
            _stats_cache = {}
    return _stats_cache


def expected_points(p_a: float, p_d: float, p_r: float) -> tuple:
    e_approved = 8.0 * p_a - 4.0 * p_d + 1.0 * p_r
    e_denied = 8.0 * p_d + 1.0 * p_r
    e_review = 8.0 * p_r + 2.0 * (p_a + p_d)
    return e_approved, e_denied, e_review


def _clamp(value: float) -> float:
    return max(CONF_MIN, min(CONF_MAX, value))


def decide(rule_path, stats=None) -> tuple:
    """Return (adjudication, confidence) for a RulePath.

    stats maps path_id -> {"dist": [pA,pD,pR], "n": int,
    "decided_label_cv_accuracy": float}. Defaults to the packaged artifact.
    """
    if stats is None:
        stats = load_stats()

    entry = stats.get(rule_path.path_id)

    if rule_path.dist is not None:
        dist = rule_path.dist
    elif entry is not None:
        dist = tuple(entry["dist"])
    else:
        # Unseen empirical path: uniform prior -> expected points favor the
        # conservative NEEDS_REVIEW hedge.
        dist = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    scores = expected_points(*dist)
    best = max(scores)
    # On exact ties prefer NEEDS_REVIEW (index 2), then DENIED over APPROVED
    # (never break a tie toward the -4 false-approval risk).
    for idx in (2, 1, 0):
        if scores[idx] == best:
            adjudication = _LABELS[idx]
            break

    if adjudication == "APPROVED" and dist[1] > FA_GUARD_MAX_PD:
        adjudication = "NEEDS_REVIEW"

    # tools/fit_stats.py applies the same guard when measuring out-of-fold
    # accuracy, so the stored accuracy describes the guarded decision too.
    if entry is not None and entry.get("decided_label_cv_accuracy") is not None:
        confidence = entry["decided_label_cv_accuracy"]
    else:
        confidence = dist[_LABELS.index(adjudication)]

    return adjudication, _clamp(confidence)
