"""Adversarial probes of `circuit_check.cache`: an edit that can change a target's verdict changes its key."""
from __future__ import annotations


#: `targets.py` with two `_model_roots` entries of one label
DUP_TARGETS = """
def _model_roots():
    return [
        ("SynOther_v1", lambda: "SynOther_v1"),
        ("Dup_v1", lambda: "first"),
        ("Dup_v1", lambda: "second"),
    ]
"""


def test_an_edit_to_either_of_two_same_label_entries_changes_a_unit():
    """`_units_of` keys each `_model_roots` entry by its label, so of two entries with one label only the last is digested,
    though `catalog` runs both (each under that one unit, `_produce`)."""
    from circuit_check import cache as CC

    units = CC._units_of(DUP_TARGETS.lstrip("\n"))[0]
    edited = CC._units_of(DUP_TARGETS.lstrip("\n").replace('lambda: "first"', 'lambda: "edited"'))[0]
    assert edited != units, "an edit to the first Dup_v1 entry changed no unit: every target that holds it keeps its key"
