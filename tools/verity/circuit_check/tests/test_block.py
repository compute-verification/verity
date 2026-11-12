"""circuit-check's ``block`` (`checks.block_range`): a Definition checked whole fits the Flock block `compose` would place it
in as one unit, its rows' hash slots and mask beside its own slot, within Flock's proved range of 2^`K_MAX` bits, read off
the Definition before any lowering; past it, it fails by name with the breakdown and the fix, unless `BLOCK_HELD` says
why not, and a held one that fits fails as stale, as does one the whole catalog no longer checks.  A Call its module cuts
(``CUTS``, `checks.cut_of`) is held to the range by each unit of its cut instead."""
from __future__ import annotations

import sys

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


def _rows():
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.types import Array, Value

    return bind(CompositeDefinition("CircuitCheckBlockRows", 1, (), lambda S: ((("x", Array(1 << 17, Value(16))),), Array(1 << 17, Value(16))),
                                    lambda B, S, x: x, register=False))


def test_rows_that_push_a_fitting_slot_past_the_block_fail():
    """A gate-free unit copying a row of 2^17 words: its own slot of 2^23 cells fits 2^`K_MAX` (the check before it counted
    only that, and passed it), but its two 256 KiB rows take 1,366 sha512x3 slots, so the block `compose` would place it in
    is 2^29 bits and ``block`` fails by name, with the breakdown."""
    from verity_flock import circuit as FC

    fn = _rows()
    assert FC.past_range(FC.unit_witness(fn)) is None
    F = CK.Findings()
    b = CK.block_range(fn, F)
    assert (b["unit_log"], b["sha512x3"], b["k_log"]) == (23, 1366, 29)
    (f,) = F.failures
    assert f["name"].startswith("block/past-range CircuitCheckBlockRows_v1 the unit needs a block of 2^29 bits (367,263,744): "
                                "its slot of 2^23 (4,194,305 cells for 0 ANDs and its ports, read off its Definition), its 2 "
                                "rows' 1366 sha512x3 and 2 hm96 slots of 2^18, and the mask")
    assert (f["bits"], f["rows"], f["hm96"]) == (367_263_744, 2, 2)


def test_the_held_list_only_shrinks():
    """`block_held.HELD` is at most `block_held.MAX` long, and each entry says why."""
    from circuit_check import block_held as BH

    assert len(BH.HELD) <= BH.MAX and all(why.strip() for why in BH.HELD.values())


def test_a_held_definition_is_reported_held_and_a_held_one_that_fits_is_stale(monkeypatch):
    """A Definition `block_held.HELD` names: past the block, the report holds its ``block/past-range`` failure with why and
    doesn't count it; fitting, the report fails it ``block/stale-hold``, so the list shrinks with each fix.  The target's own
    entry is the same either way: no target's check reads the list."""
    from circuit_check import block_held as BH
    from verity_flock import circuit as FC

    fn = _toy()
    monkeypatch.setitem(BH.HELD, fn.id, "a toy")
    entry = CK.check_definition(fn, OPTS, as_call=False)
    s = CK.summarize([entry])
    assert "block_held" not in s
    (name,) = [n for n in s["failure_names"] if n.startswith("block/")]
    assert name.startswith("block/stale-hold CircuitCheckBlock_v1 its block of 2^20 bits fits 2^27, but block_held.HELD holds it "
                           "(a toy): delete its entry")
    monkeypatch.setattr(FC, "K_MAX", 19)
    F = CK.Findings()
    past = {**entry, "block": CK.block_range(fn, F), "failures": F.failures}
    assert [f["code"] for f in F.failures] == ["past-range"]
    s = CK.summarize([past])
    assert s["block_held"] == {fn.id: "a toy"} and not [n for n in s["failure_names"] if n.startswith("block/")]
    monkeypatch.delitem(BH.HELD, fn.id)
    assert [n for n in CK.summarize([past])["failure_names"] if n.startswith("block/past-range CircuitCheckBlock_v1")]


def test_a_cut_call_is_held_to_the_block_by_each_unit(monkeypatch):
    """A Call whose module's ``CUTS`` names its family: the whole (2^29 bits) needs no block, each unit of its cut does; the
    entry carries the largest unit's block, the whole's and each unit's with its count.  A unit past the range fails naming
    the Call and the unit; a held Call whose units fit is stale."""
    from circuit_check import block_held as BH
    from verity_flock import circuit as FC

    whole, unit = _rows(), _toy()
    assert CK.cut_of(whole) is None
    monkeypatch.setattr(sys.modules[__name__], "CUTS", {"CircuitCheckBlockRows_v1": lambda t: [(unit, 3)]}, raising=False)
    assert CK.cut_of(whole) == [(unit, 3)] and CK.cut_of(unit) is None
    F = CK.Findings()
    b = CK.block_range(whole, F)
    assert not F.failures
    assert (b["k_log"], b["whole"]["k_log"]) == (20, 29)
    assert b["cut"] == [{"unit": "CircuitCheckBlock_v1", "count": 3, "ands": b["cut"][0]["ands"], "bits": b["bits"], "k_log": 20}]
    monkeypatch.setitem(BH.HELD, whole.id, "a toy")
    (name,) = [n for n in CK.summarize([{"definition": whole.id, "block": b, "failures": []}])["failure_names"]]
    assert name.startswith("block/stale-hold CircuitCheckBlockRows_v1 each unit of its cut fits 2^27 (the largest 2^20 bits), "
                           "but block_held.HELD holds it (a toy)")
    monkeypatch.setattr(FC, "K_MAX", 19)
    F = CK.Findings()
    CK.block_range(whole, F)
    (f,) = F.failures
    assert f["name"].startswith("block/past-range CircuitCheckBlockRows_v1 its cut's unit CircuitCheckBlock_v1: the unit needs "
                                "a block of 2^20 bits")


def test_a_held_id_the_catalog_no_longer_checks_fails(monkeypatch):
    """Over the whole catalog (an unsharded ``--all``, or every shard of one combined), a held id no entry names fails
    ``block/absent-hold``, so a Definition renamed or removed takes its entry with it; a report over some targets can't tell,
    and fails nothing for it."""
    from circuit_check import block_held as BH

    fn = _toy()
    monkeypatch.setattr(BH, "HELD", {fn.id: "a toy", "CircuitCheckGone_v1": "gone"})
    entry = CK.check_definition(fn, OPTS, as_call=False)
    assert not [n for n in CK.summarize([entry])["failure_names"] if "absent-hold" in n]
    names = CK.summarize([entry], catalog=True)["failure_names"]
    assert [n for n in names if "absent-hold" in n] == [
        "block/absent-hold CircuitCheckGone_v1 block_held.HELD holds it (gone), but the catalog checks no Definition of that id "
        "any longer: delete its entry (tools/verity/circuit_check/src/circuit_check/block_held.py) and lower MAX"]
    assert CK.report([entry], OPTS, catalog=True)["summary"]["failure_names"] == names


def test_sparse_mla_is_held_to_the_block_by_its_head_chain():
    """GLM-5.3's `SparseMla_v1` at its circuit-check binding (T = 65, the served heads and widths) is past the block whole
    (2^30 bits) and cut by `sparse_mla.CUTS` into each head's blocks and end, whose largest, a full block, fits."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import sparse_mla as M

    F = CK.Findings()
    b = CK.block_range(bind(M.SparseMla, T=65, **M.SPARSE_MLA_GLM53), F)
    assert not F.failures and b["whole"]["k_log"] == 30 and b["k_log"] == 27
    assert [(u["unit"].partition("{")[0], u["count"]) for u in b["cut"]] == [
        ("SparseMlaHeadBlock_v1", 8), ("SparseMlaHeadBlock_v1", 8), ("SparseMlaHeadEnd_v1", 8)]
