"""`credit.passes_distinct` (`Properties.Pouw.LiveSalt.PassesDistinct`): Pearl-C passes credited together hold
each call index of their epoch (their salt) once, each below `pearl_c.CALL_RANGE`."""
from __future__ import annotations

from verity.core.protocols.pouw import credit as Cr
from verity.ml.protocols.pouw.schemes import pearl_c as PC

SALT, OTHER = "ab" * 32, "cd" * 32


def _pass(salt: str, *indices: int) -> dict:
    return {"salt": salt, "matmuls": [{"index": i} for i in indices]}


def test_an_epochs_passes_numbered_on_are_distinct():
    assert Cr.passes_distinct([_pass(SALT, 0, 1), _pass(SALT, 2, 3), _pass(SALT, 4, 5)]) is None
    assert Cr.passes_distinct([_pass(SALT, 0, 1), _pass(OTHER, 0, 1)]) is None      # two epochs share no seed
    assert Cr.passes_distinct([]) is None and Cr.passes_distinct([_pass(SALT)]) is None


def test_a_call_index_two_passes_of_one_epoch_hold_is_refused():
    code, why = Cr.passes_distinct([_pass(SALT, 0, 1), _pass(SALT, 1, 2)])
    assert code == Cr.REPEATED_CALL == "repeated-call" and "passes 0 and 1 both hold call index 1" in why
    assert Cr.passes_distinct([_pass(SALT, 0, 1), _pass(SALT, 0, 1)])[0] == "repeated-call"   # a pass listed twice
    code, why = Cr.passes_distinct([_pass(SALT, 3, 3)])
    assert code == "repeated-call" and "pass 0 holds call index 3 twice" in why


def test_a_call_index_at_2_to_the_32_is_refused():
    assert PC.CALL_RANGE == 1 << 32
    assert Cr.passes_distinct([_pass(SALT, PC.CALL_RANGE - 1)]) is None
    for bad in (PC.CALL_RANGE, PC.CALL_RANGE + 7, -1, 1.0, "0", True, None):
        code, why = Cr.passes_distinct([_pass(SALT, 0), _pass(SALT, bad)])
        assert code == Cr.CALL_RANGE_CODE == "call-range" and "end the epoch" in why, bad


def test_a_pass_that_is_no_manifest_is_refused():
    for bad in (None, [], {"salt": SALT}, {"matmuls": []}, {"salt": 1, "matmuls": []}, {"salt": SALT, "matmuls": [1]}):
        assert Cr.passes_distinct([bad])[0] == "manifest", bad
