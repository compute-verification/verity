"""circuit-check's ``block`` (`checks.block_range`): a Definition checked whole fits the Flock block `compose` would place it
in as one unit, its rows' hash slots and mask beside its own slot, within Flock's proved range of 2^`K_MAX` bits, read off
the Definition before any lowering; past it, it fails by name with the breakdown and the fix, unless `BLOCK_HELD` says
why not, and a held one that fits fails as stale."""
from __future__ import annotations

from circuit_check import checks as CK

OPTS = CK.Options()


def _toy():
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.types import Array, Value
    from verity.ml.definitions.registry import prims as P

    return bind(CompositeDefinition("CircuitCheckBlock", 1, (), lambda S: ((("x", Array(2, Value(32))),), Value(32)),
                                    lambda B, S, x: B.call(P.F32Add, x[0], x[1]), register=False))


def test_a_unit_past_c_flocks_range_is_a_block_failure(monkeypatch):
    """A Definition checked whole: its block (`verity_flock.circuit.unit_block`) fits Flock's proved range of 2^`K_MAX`,
    and past it fails by name with the fix, as `compose` would refuse it.  One checked as a template's proof blocks has its
    block recorded, not failed."""
    from verity_flock import circuit as FC

    fn = _toy()
    b = CK.check_definition(fn, OPTS, as_call=False)["block"]
    assert b["applies"] and 0 < b["ands"] and b["port_bits"] == 96 and b["rows"] == 2 and b["k_log"] <= FC.K_MAX
    assert not [f for f in CK.check_definition(fn, OPTS, as_call=False)["failures"] if f["check"] == "block"]
    monkeypatch.setattr(FC, "K_MAX", b["k_log"] - 1)
    (f,) = [f for f in CK.check_definition(fn, OPTS, as_call=False)["failures"] if f["check"] == "block"]
    assert f["name"].startswith(f"block/past-range CircuitCheckBlock_v1 the unit needs a block of 2^{b['k_log']} bits")
    assert f["name"].endswith("split the unit (e.g. PoUW's lanes form, verity/core/protocols/pouw/leaf_lanes.py), or extend "
                              "the proved range (proofs)")
    assert (f["ands"], f["k_log"], f["sha512x3"]) == (b["ands"], b["k_log"], b["sha512x3"])
    F = CK.Findings()
    assert not CK.block_range(fn, F, applies=False)["applies"] and not F.failures


def test_rows_that_push_a_fitting_slot_past_the_block_fail():
    """A gate-free unit copying a row of 2^17 words: its own slot of 2^23 cells fits 2^`K_MAX` (the check before it counted
    only that, and passed it), but its two 256 KiB rows take 1,366 sha512x3 slots, so the block `compose` would place it in
    is 2^29 bits and ``block`` fails by name, with the breakdown."""
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.types import Array, Value
    from verity_flock import circuit as FC

    fn = bind(CompositeDefinition("CircuitCheckBlockRows", 1, (), lambda S: ((("x", Array(1 << 17, Value(16))),), Array(1 << 17, Value(16))),
                                  lambda B, S, x: x, register=False))
    assert FC.past_range(FC.unit_witness(fn)) is None
    F = CK.Findings()
    b = CK.block_range(fn, F)
    assert (b["unit_log"], b["sha512x3"], b["k_log"]) == (23, 1366, 29)
    (f,) = F.failures
    assert f["name"].startswith("block/past-range CircuitCheckBlockRows_v1 the unit needs a block of 2^29 bits (367,263,744): "
                                "its slot of 2^23 (4,194,305 cells for 0 ANDs and its ports, read off its Definition), its 2 "
                                "rows' 1366 sha512x3 and 2 hm96 slots of 2^18, and the mask")
    assert (f["bits"], f["rows"], f["hm96"]) == (367_263_744, 2, 2)


def test_a_held_definition_is_recorded_and_a_held_one_that_fits_is_stale(monkeypatch):
    """`BLOCK_HELD` names a Definition past the block and why: its block is recorded with the reason, not failed.  A held
    Definition whose block fits fails ``block/stale-hold``, so the list shrinks with each fix; `BLOCK_HELD_MAX` bounds it."""
    from verity_flock import circuit as FC

    assert len(CK.BLOCK_HELD) <= CK.BLOCK_HELD_MAX
    fn = _toy()
    monkeypatch.setitem(CK.BLOCK_HELD, fn.id, "a toy")
    F = CK.Findings()
    assert CK.unit_block(fn, F)["held"] == "a toy"
    (f,) = F.failures
    assert f["name"].startswith("block/stale-hold CircuitCheckBlock_v1 its block of 2^20 bits fits 2^27, but it is held (a toy)")
    monkeypatch.setattr(FC, "K_MAX", 19)
    F = CK.Findings()
    assert CK.unit_block(fn, F)["held"] == "a toy" and not F.failures
