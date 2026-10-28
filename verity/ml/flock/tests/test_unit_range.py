"""`circuit.refuse_past_range`: a unit whose Definition alone needs more witness bits than C-Flock's proved range of
2^`K_MAX` is refused by `UnitStager` before it is lowered, read off the Definition (`ir_lower.definition_ands`); one in
range is staged as before. The count is held to the flat lowering's ANDs on staged units, and compose refuses what this
refuses unless the lowering lays out fewer than half the witness bits counted."""
from __future__ import annotations

import pytest
from test_class_statement_unit_stager import STATEMENTS, TOY, TOY_ROWS
from verity.ml.circuits.boolean import AND, BIT, and_gate
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS


def wide(log: int):
    """A unit of 2^(log + 1) - 1 ANDs on 2 input bits and 1 output bit: 2^log two-input Ands on one shared input, then one
    And over them, so its slot is 2^(log + 2); two body nodes, built and counted in milliseconds at any size."""
    n = 1 << log
    return bind(CompositeDefinition(f"UnitRangeWide{log}", 1, (), lambda S: ((("x", Array(2, BIT)),), BIT),
                                    lambda B, S, x: B.call(and_gate(n), B.batch(AND, x, axes=(None,), n=n)), register=False))


def test_a_unit_past_the_range_is_refused_before_it_is_lowered(monkeypatch):
    """A unit of 2^27 - 1 ANDs needs a slot of 2^28 witness bits, past C-Flock's 2^27: `UnitStager` refuses it by name,
    with the fix, and never lowers it (a flat unit of that size is about 80 GB of Python objects)."""
    def lowered(*_a, **_k):
        raise AssertionError("the unit was lowered before its range was checked")

    monkeypatch.setattr(CS, "unit_lowering", lowered)
    d = wide(FC.K_MAX - 1)
    assert FC.unit_witness(d) == {"ands": (1 << FC.K_MAX) - 1, "port_bits": 3, "k": FC.K_MAX + 1, "k_max": FC.K_MAX}
    with pytest.raises(ValueError, match=r"^unit/wide: the unit needs a slot of 2\^28 witness bits \(134,217,727 ANDs and 3 port "
                                         r"bits, read off its Definition\); C-Flock's proved range is 2\^27 \(Flock/Statement\.lean "
                                         r"Stmt\.InRange\): split the unit \(e\.g\. PoUW's lanes form, "
                                         r"verity/core/protocols/pouw/leaf_lanes\.py\) or extend the proved range \(proofs\)$"):
        CS.UnitStager(d, "unit/wide", [("x", V2, 2, ["x"])], [(V2, (0,))])


def test_the_range_is_k_max_read_at_the_check():
    """At a slot of exactly 2^`K_MAX` the unit is in range, one power past it is not; the limit is `circuit.K_MAX` itself."""
    assert FC.refuse_past_range(wide(FC.K_MAX - 2), "unit/at")["k"] == FC.K_MAX
    with pytest.raises(ValueError, match=r"^unit/past: the unit needs a slot of 2\^28 witness bits"):
        FC.refuse_past_range(wide(FC.K_MAX - 1), "unit/past")


def test_a_unit_in_range_is_staged(monkeypatch):
    """The toy unit is in range and staged as before; with `K_MAX` one below its slot it is refused, before lowering."""
    w = FC.refuse_past_range(TOY, "unit/stager-toy")
    st = CS.UnitStager(TOY, "unit/stager-toy", TOY_ROWS, [(V2, (0,))])
    assert w["k"] <= st.meta["k_log"] <= FC.K_MAX
    monkeypatch.setattr(FC, "K_MAX", w["k"] - 1)
    monkeypatch.setattr(CS, "unit_lowering", lambda *a, **k: pytest.fail("lowered"))
    with pytest.raises(ValueError, match=rf"^unit/stager-toy: the unit needs a slot of 2\^{w['k']} witness bits"):
        CS.UnitStager(TOY, "unit/stager-toy", TOY_ROWS, [(V2, (0,))])


@pytest.mark.parametrize("which", list(STATEMENTS))
def test_the_count_is_the_flat_lowerings(which):
    """On `UnitStager`'s statements (a Boolean toy, two PoUW tile units, the anchors' stand-in) the Definition's count is the
    flat lowering's ANDs, exactly on the toy and within 2% where the lowering drops ANDs a constant or no reader leaves, and
    the lowering lays out more than half the witness bits counted, so compose would refuse whatever this refuses."""
    (d, name, in_rows, _out), _rows, kw = STATEMENTS[which]()
    w = FC.unit_witness(d)
    low, _ = CS.unit_lowering(d, name, [(r, ps) for r, _s, _b, ps in in_rows], tuple(kw.get("public_inputs") or {}))
    ands = low.layout.counts["ands"]
    assert ands <= w["ands"] <= ands * 1.02 and (which != "toy" or w["ands"] == ands)
    assert 2 * low.layout.useful > w["ands"] + w["port_bits"]
