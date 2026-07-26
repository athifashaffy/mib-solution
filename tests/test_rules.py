"""Cascade-order tests for mib.rules.path_of and flag inference.

All records are synthetic; no PDFs or fitted stats needed.
"""
from mib.resolve import CaseRecord, EvidenceSummary
from mib.rules import infer_flags, path_of


def make_record(fields=None, evidence=None, risk_flags=None):
    base_fields = {
        "case_id": "MIB-000001",
        "applicant_name": "Zed Zarnax",
        "species_code": "ORION_GRAYS",
        "home_world": "Kepler-186f",
        "visa_class": "XW-1",
        "sponsor_id": "SPN-1234",
        "arrival_date": "2026-04-01",
        "declared_purpose": "research",
        "fee_status": "paid",
    }
    base_fields.update(fields or {})
    return CaseRecord(
        fields=base_fields,
        evidence=evidence or EvidenceSummary(),
        risk_flags=set(risk_flags or ()),
    )


# --- flag inference ---------------------------------------------------------

def test_embargoed_world_infers_planetary_embargo():
    rec = make_record(fields={"home_world": "TRAPPIST-1e"})
    assert "planetary_embargo" in infer_flags(rec)


def test_eris_relay_infers_planetary_embargo():
    rec = make_record(fields={"home_world": "Eris Relay"})
    assert "planetary_embargo" in infer_flags(rec)


def test_normal_world_infers_nothing():
    rec = make_record()
    assert infer_flags(rec) == set()


def test_inference_preserves_existing_flags():
    rec = make_record(fields={"home_world": "TRAPPIST-1e"},
                      risk_flags={"illegible_biometrics"})
    assert infer_flags(rec) == {"illegible_biometrics", "planetary_embargo"}


# --- cascade order ----------------------------------------------------------

def test_note_verdict_beats_everything():
    # DENIED-looking case, but the adjudicator note says APPROVED.
    rec = make_record(fields={"visa_class": "TRANSIT-7", "fee_status": "unpaid"},
                      evidence=EvidenceSummary(note_verdict="APPROVED"),
                      risk_flags={"biohazard_red"})
    rp = path_of(rec)
    assert rp.path_id == "note:APPROVED"
    assert rp.dist == (1.0, 0.0, 0.0)


def test_disqualifying_flag_denies():
    rec = make_record(risk_flags={"biohazard_red"})
    rp = path_of(rec)
    assert rp.path_id == "disq"
    assert rp.dist == (0.0, 1.0, 0.0)


def test_disq_beats_transit():
    rec = make_record(fields={"visa_class": "TRANSIT-7"},
                      risk_flags={"active_warrant"})
    assert path_of(rec).path_id == "disq"


def test_transit_denies():
    rec = make_record(fields={"visa_class": "TRANSIT-7"})
    rp = path_of(rec)
    assert rp.path_id == "transit"
    assert rp.dist == (0.0, 1.0, 0.0)


def test_fee_unpaid_denies():
    rec = make_record(fields={"fee_status": "unpaid"})
    rp = path_of(rec)
    assert rp.path_id == "unpaid"
    assert rp.dist == (0.0, 1.0, 0.0)


def test_fee_unknown_needs_review():
    rec = make_record(fields={"fee_status": "unknown"})
    rp = path_of(rec)
    assert rp.path_id == "feeunk"
    assert rp.dist == (0.0, 0.0, 1.0)


def test_fee_missing_is_not_deterministic():
    # Extraction failure (None) is not the same as an extracted "unknown";
    # it falls through to an empirical path with a quality suffix counting
    # the missing field.
    rec = make_record(fields={"fee_status": None})
    rp = path_of(rec)
    assert rp.dist is None
    assert "fee:missing" in rp.path_id
    assert ";m1;" in rp.path_id


def test_revoked_sponsor_denies_for_non_dip():
    rec = make_record(fields={"sponsor_id": "SPN-7331"})
    rp = path_of(rec)
    assert rp.path_id == "revoked"
    assert rp.dist == (0.0, 1.0, 0.0)


def test_revoked_sponsor_ignored_for_dip1():
    rec = make_record(fields={"sponsor_id": "SPN-7331", "visa_class": "DIP-1"})
    rp = path_of(rec)
    assert rp.path_id != "revoked"


def test_all_six_revoked_sponsors():
    for spn in ("SPN-0007", "SPN-0139", "SPN-4040", "SPN-7331", "SPN-2718", "SPN-9090"):
        rec = make_record(fields={"sponsor_id": spn})
        assert path_of(rec).path_id == "revoked", spn


def test_missing_sponsor_non_dip_is_empirical():
    rec = make_record(fields={"sponsor_id": None})
    rp = path_of(rec)
    assert rp.path_id.startswith("nosponsor;")
    assert rp.dist is None


def test_missing_sponsor_dip1_ok():
    rec = make_record(fields={"sponsor_id": None, "visa_class": "DIP-1"})
    assert not path_of(rec).path_id.startswith("nosponsor")


def test_missing_arrival_date_is_empirical():
    rec = make_record(fields={"arrival_date": None})
    rp = path_of(rec)
    assert rp.path_id.startswith("nodate;")
    assert rp.dist is None


def test_hidden_only_arrival_date_is_deterministic_review():
    rec = make_record(
        evidence=EvidenceSummary(hidden_only_fields={"arrival_date"}))
    rp = path_of(rec)
    assert rp.path_id == "nodate-hidden"
    assert rp.dist == (0.0, 0.0, 1.0)


def test_stale_arrival_denies_for_non_dip():
    rec = make_record(fields={"arrival_date": "2025-12-01"})
    rp = path_of(rec)
    assert rp.path_id == "stale"
    assert rp.dist == (0.0, 1.0, 0.0)


def test_stale_boundary_not_stale_on_cutoff():
    rec = make_record(fields={"arrival_date": "2026-01-08"})
    assert path_of(rec).path_id != "stale"


def test_old_dip1_is_not_stale():
    rec = make_record(fields={"arrival_date": "2025-12-01", "visa_class": "DIP-1",
                              "sponsor_id": None})
    assert path_of(rec).path_id != "stale"


def test_clean_case_gets_empirical_path():
    rec = make_record()
    rp = path_of(rec)
    assert rp.dist is None
    assert rp.path_id == "flags:none;fee:paid;visa:std;q:t;m0;b0;g0"


def test_review_flags_go_to_empirical_path():
    rec = make_record(risk_flags={"illegible_biometrics"})
    rp = path_of(rec)
    assert rp.dist is None
    assert rp.path_id == "flags:illegible_biometrics;fee:paid;visa:std;q:t;m0;b0;g0"


def test_flag_profile_is_sorted_in_path_id():
    rec = make_record(risk_flags={"sponsor_mismatch", "illegible_biometrics"})
    rp = path_of(rec)
    assert "flags:illegible_biometrics|sponsor_mismatch" in rp.path_id


def test_waived_dip_path_distinct_from_waived_std():
    dip = make_record(fields={"fee_status": "waived", "visa_class": "DIP-1"})
    std = make_record(fields={"fee_status": "waived"})
    assert path_of(dip).path_id != path_of(std).path_id


def test_inferred_embargo_reaches_cascade():
    # Embargoed home world with no explicit flag must still deny via "disq".
    rec = make_record(fields={"home_world": "Eris Relay"})
    assert path_of(rec).path_id == "disq"
