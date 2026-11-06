"""circuit-check's ``block`` (`checks.block_range`): a Definition staged whole as one C-Flock unit fits C-Flock's proved range
of 2^`K_MAX` witness bits, read off the Definition before any lowering, and past it fails by name with the fix."""
from __future__ import annotations

from circuit_check import checks as CK

OPTS = CK.Options()


def test_a_unit_past_c_flocks_range_is_a_block_failure(monkeypatch):
    """A Definition staged whole as one C-Flock unit (one `UNIT_OWNERS` owns, PoUW's): its ANDs and port bits, read off it
    (`verity_flock.circuit.unit_witness`), fit C-Flock's proved range of 2^`K_MAX`, and past it fail by name with the fix, as
    `UnitStager` refuses before lowering.  Any other Definition, and one checked as a template's proof blocks, has its count
    recorded, not failed."""
    from verity_flock import circuit as FC
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.types import Array, Value
    from verity.ml.definitions.registry import prims as P

    fn = bind(CompositeDefinition("CircuitCheckBlock", 1, (), lambda S: ((("x", Array(2, Value(32))),), Value(32)),
                                  lambda B, S, x: B.call(P.F32Add, x[0], x[1]), register=False))
    w = CK.check_definition(fn, OPTS, as_call=False)["block"]
    assert not w["applies"] and 0 < w["ands"] and w["port_bits"] == 96 and w["k"] <= w["k_max"] == FC.K_MAX
    monkeypatch.setattr(FC, "K_MAX", w["k"] - 1)
    assert not [f for f in CK.check_definition(fn, OPTS, as_call=False)["failures"] if f["check"] == "block"]
    monkeypatch.setattr(CK, "UNIT_OWNERS", (__name__,))
    e = CK.check_definition(fn, OPTS, as_call=False)
    assert e["block"]["applies"]
    (f,) = [f for f in e["failures"] if f["check"] == "block"]
    assert f["name"].startswith(f"block/past-range CircuitCheckBlock_v1 the unit needs a slot of 2^{w['k']} witness bits")
    assert f["name"].endswith("split the unit (e.g. PoUW's lanes form, verity/core/protocols/pouw/leaf_lanes.py) or extend "
                              "the proved range (proofs)")
    assert (f["ands"], f["k_max"]) == (w["ands"], w["k"] - 1)
    F = CK.Findings()
    assert not CK.block_range(fn, F, applies=False)["applies"] and not F.failures
