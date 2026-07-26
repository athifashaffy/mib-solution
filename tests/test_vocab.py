"""Tests for mib.vocab: closed vocabularies + edit-distance snapping."""
from mib.vocab import VOCAB, snap, snap_name


def test_vocab_has_expected_kinds():
    for kind in ("species", "world", "purpose", "visa", "fee", "flag", "first_names", "last_names"):
        assert kind in VOCAB
        assert len(VOCAB[kind]) > 0


def test_vocab_kind_sizes():
    assert len(VOCAB["species"]) == 12
    assert len(VOCAB["world"]) == 13
    assert len(VOCAB["purpose"]) == 10
    assert len(VOCAB["visa"]) == 5
    assert len(VOCAB["fee"]) == 4
    assert len(VOCAB["flag"]) == 8
    assert len(VOCAB["first_names"]) == 144
    assert len(VOCAB["last_names"]) == 144


def test_snap_exact_match_case_insensitive():
    assert snap("jovian_gasform", "species") == "JOVIAN_GASFORM"
    assert snap("JOVIAN_GASFORM", "species") == "JOVIAN_GASFORM"
    assert snap("proxima-b", "world") == "Proxima-b"


def test_snap_species_typo_within_threshold():
    assert snap("JOVIAN_GAS M", "species") == "JOVIAN_GASFORM"


def test_snap_visa_typo_within_threshold():
    assert snap("TRANS1T-7", "visa") == "TRANSIT-7"


def test_snap_world_typo_within_threshold():
    assert snap("Prox1ma-b", "world") == "Proxima-b"


def test_snap_name_typo_within_threshold():
    assert snap_name("Tektari So1ix") == "Tektari Solix"


def test_snap_garbage_returns_none():
    assert snap("xxxxxxxxxxxxxxxxxxxx", "species") is None
    assert snap("zzz", "visa") is None
    assert snap("completely unrelated purpose text", "purpose") is None


def test_snap_none_and_empty_input():
    assert snap(None, "species") is None
    assert snap("", "species") is None
    assert snap("   ", "species") is None


def test_snap_whitespace_collapsed_and_stripped():
    assert snap("  jovian_gasform  ", "species") == "JOVIAN_GASFORM"
    assert snap("cultural   exchange", "purpose") == "cultural exchange"


def test_snap_unknown_kind_returns_none():
    assert snap("anything", "not_a_kind") is None


def test_snap_tie_returns_none():
    # Two single-character-distance entries from an ambiguous midpoint tie
    # at the same minimal distance -> None rather than an arbitrary pick.
    assert snap("XW", "visa") is None


def test_snap_name_wrong_token_count_returns_none():
    assert snap_name("OnlyOneToken") is None
    assert snap_name("Three Token Names") is None
    assert snap_name(None) is None
    assert snap_name("") is None


def test_snap_name_unmatched_token_returns_none():
    assert snap_name("Zzzzzzzzzz Qorquell") is None
