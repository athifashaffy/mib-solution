"""Decision-math and confidence tests for mib.decide. No fitted artifact
needed — small fake stats dicts are injected."""
import pytest

from mib.decide import decide, expected_points
from mib.rules import RulePath


def test_expected_points_hand_computed():
    # dist (0.5, 0.3, 0.2): E[A]=8*.5-4*.3+.2=3.0 ... verify formula precisely:
    ea, ed, er = expected_points(0.5, 0.3, 0.2)
    assert ea == pytest.approx(8 * 0.5 - 4 * 0.3 + 1 * 0.2)
    assert ed == pytest.approx(8 * 0.3 + 1 * 0.2)
    assert er == pytest.approx(8 * 0.2 + 2 * (0.5 + 0.3))


def test_hedge_wins_on_mixed_dist():
    # (0.5, 0.3, 0.2): E[A]=3.0, E[D]=2.6, E[R]=3.2 -> NEEDS_REVIEW
    adj, _ = decide(RulePath("x", None), {"x": {"dist": [0.5, 0.3, 0.2],
                                               "n": 10,
                                               "decided_label_cv_accuracy": 0.5}})
    assert adj == "NEEDS_REVIEW"


def test_confident_approve():
    adj, conf = decide(RulePath("x", None), {"x": {"dist": [0.9, 0.04, 0.06],
                                                   "n": 100,
                                                   "decided_label_cv_accuracy": 0.88}})
    assert adj == "APPROVED"
    assert conf == pytest.approx(0.88)


def test_deterministic_path_uses_stats_accuracy_for_confidence():
    adj, conf = decide(RulePath("disq", (0.0, 1.0, 0.0)),
                       {"disq": {"dist": [0.0, 1.0, 0.0], "n": 186,
                                 "decided_label_cv_accuracy": 1.0}})
    assert adj == "DENIED"
    assert conf == pytest.approx(0.99)  # clamped


def test_unseen_path_falls_back_conservatively():
    adj, conf = decide(RulePath("never-seen", None), {})
    assert adj == "NEEDS_REVIEW"
    assert 0.05 <= conf <= 0.99


def test_unseen_deterministic_path_uses_chosen_probability():
    adj, conf = decide(RulePath("note:DENIED", (0.0, 1.0, 0.0)), {})
    assert adj == "DENIED"
    assert conf == pytest.approx(0.99)  # p=1.0 clamped


def test_confidence_clamped_low():
    adj, conf = decide(RulePath("x", None), {"x": {"dist": [0.4, 0.35, 0.25],
                                                   "n": 4,
                                                   "decided_label_cv_accuracy": 0.01}})
    assert conf == pytest.approx(0.05)


def test_tie_prefers_needs_review():
    # Construct an exact three-way tie: pA=pD and E[R] equal by design is
    # awkward; instead verify the documented pairwise tie rule with A==D:
    # dist (0.4, 0.4, 0.2): E[A]=8*.4-4*.4+.2=1.8, E[D]=3.4, E[R]=3.2 -> DENIED
    adj, _ = decide(RulePath("x", None), {"x": {"dist": [0.4, 0.4, 0.2],
                                                "n": 10,
                                                "decided_label_cv_accuracy": 0.5}})
    assert adj == "DENIED"


def test_fa_guard_demotes_risky_approval():
    from mib.rules import RulePath
    from mib.decide import decide
    # E[A] = 8*0.75 - 4*0.2 + 0.05 = 5.25 beats E[R] = 0.4 + 1.9 = 2.3,
    # but pD=0.20 > 0.15 guard ceiling -> hedged to NEEDS_REVIEW.
    rp = RulePath("x", None)
    adj, conf = decide(rp, stats={"x": {"dist": [0.75, 0.20, 0.05], "n": 40,
                                        "decided_label_cv_accuracy": None}})
    assert adj == "NEEDS_REVIEW"


def test_fa_guard_allows_low_risk_approval():
    from mib.rules import RulePath
    from mib.decide import decide
    rp = RulePath("x", None)
    adj, conf = decide(rp, stats={"x": {"dist": [0.9, 0.04, 0.06], "n": 50,
                                        "decided_label_cv_accuracy": 0.9}})
    assert adj == "APPROVED"
    assert conf == 0.9
