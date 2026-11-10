"""`circuit.refuse_past_range`: a unit whose Definition alone needs more witness bits than Flock's proved range of
2^`K_MAX` is refused by `UnitStager` before it is lowered, read off the Definition (`ir_lower.definition_ands`); one in
range is staged as before. The count is held to the flat lowering's ANDs on staged units, and compose refuses what this
refuses unless the lowering lays out fewer than half the witness bits counted. `circuit.unit_block`, the whole block
(rows, mask and the unit's slot) read off the Definition, is held to the block `compose` places a staged unit in."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_class_statement_unit_stager import STATEMENTS, TOY, TOY_ROWS
from verity.ml.circuits.boolean import AND, BIT, and_gate
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import ir_lower as IL


def wide(log: int):
    """A unit of 2^(log + 1) - 1 ANDs on 2 input bits and 1 output bit: 2^log two-input Ands on one shared input, then one
    And over them, so its slot is 2^(log + 2); two body nodes, built and counted in milliseconds at any size."""
    n = 1 << log
    return bind(CompositeDefinition(f"UnitRangeWide{log}", 1, (), lambda S: ((("x", Array(2, BIT)),), BIT),
                                    lambda B, S, x: B.call(and_gate(n), B.batch(AND, x, axes=(None,), n=n)), register=False))


def test_a_unit_past_the_range_is_refused_before_it_is_lowered(monkeypatch):
    """A unit of 2^27 - 1 ANDs needs a slot of 2^28 witness bits, past Flock's 2^27: `UnitStager` refuses it by name,
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


def test_a_piece_lowered_only_on_a_constant_operand_is_counted_by_its_stated_ands(monkeypatch):
    """A piece that can't be lowered on free operands (PoUW's scratch rotations by a constant count) counts the ``ands`` it
    states, under a tracer's wrapper too (`boolean_export`'s sets ``__wrapped__``), and is never lowered on free ones."""
    def piece(C, *ops):
        raise NotImplementedError("lowered on free operands")

    piece.ands = 7
    traced = lambda *a: piece(*a)  # noqa: E731
    traced.__wrapped__ = piece
    monkeypatch.setattr(IL, "_PIECE_ANDS", {})
    monkeypatch.setitem(IL.PIECES, "UnitRangeConstOnly_v1", traced)
    assert IL._piece_ands(SimpleNamespace(id="UnitRangeConstOnly_v1")) == 7


def test_a_unit_holding_the_top_p_keep_word_is_sized_by_its_composition():
    """The top-p keep word has no piece but counts its composition (`topp_word.ands`), so a unit holding it is sized off its
    Definition: in range at V = 16, and at V = 256 refused by name, before anything lowers it."""
    from verity.ml.definitions.registry import sampling as SM
    from verity_flock import topp_word as TW

    w = FC.unit_witness(SM.topp_mask_word(16))
    assert w["ands"] == TW.ands(16) and w["k"] <= FC.K_MAX
    with pytest.raises(ValueError, match=r"^unit/topp: the unit needs a slot of 2\^29 witness bits \(372,127,590 ANDs"):
        FC.refuse_past_range(SM.topp_mask_word(256), "unit/topp")


def test_the_block_read_off_a_definition_is_composes():
    """Staged on its own rows (a bit row per parameter, one per return field, as `verity.ml.windows.interface.unit_rows`
    gives them), a unit's block is the one `circuit.unit_block` reads off its Definition: the cells its lowering lays out,
    its slot, its sha512x3 and hm96 slots, and its k_log, which compose raises only by packing more units (up to 2^22)."""
    d = bind(definition("UnitRangeOwnRows", 1, lambda C, a, w: [C.AND(w[i], a[i % 32]) for i in range(1, 2048)],
                        (("a", Array(32, BIT)), ("w", Array(2048, BIT))), Array(2047, BIT), register=False))
    rows = [("a", V2, 32, ["a"]), ("w", V2, 2048, ["w"])]
    low, _ = CS.unit_lowering(d, "unit/own-rows", [(r, ps) for r, _s, _b, ps in rows])
    meta = CS.UnitStager(d, "unit/own-rows", rows, [(V2, (0,))]).meta
    b = FC.unit_block(d)
    assert (b["useful"], b["unit_log"]) == (low.layout.useful, meta["unit_log"]) == (8193, 14)
    assert (b["rows"], b["sha512x3"], b["hm96"]) == (3, meta["per_vu"][FC.SHA_NET], meta["per_vu"][FC.HM_NET])
    assert b["k_log"] <= meta["k_log"] <= max(b["k_log"], 22) and FC.past_block(b) is None


def test_rows_push_a_unit_whose_slot_fits_past_the_block():
    """A gate-free unit copying a row of 2^17 words: its slot is 2^23, in range, but its two 256 KiB rows take 1,366 sha512x3
    slots, so its block is 2^29 and `past_block` names it with the breakdown, where `past_range` passes it."""
    from verity.ml.circuits.types import Value

    d = bind(CompositeDefinition("UnitRangeRows", 1, (), lambda S: ((("x", Array(1 << 17, Value(16))),), Array(1 << 17, Value(16))),
                                 lambda B, S, x: x, register=False))
    b = FC.unit_block(d)
    assert (b["ands"], b["unit_log"], b["sha512x3"], b["hm96"], b["k_log"]) == (0, 23, 1366, 2, 29)
    assert FC.past_range(b) is None
    assert FC.past_block(b).startswith("the unit needs a block of 2^29 bits (367,263,744): its slot of 2^23 (4,194,305 cells "
                                       "for 0 ANDs and its ports, read off its Definition), its 2 rows' 1366 sha512x3 and 2 "
                                       "hm96 slots")


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
