"""`verity.circuits.supplied` (PROTOCOL.md §5.4): an `inputs` entry's supplied mark holds of its nodes for every row a
verifier makes today (the storage session's round coin, header and context, the storage setup's base and slot, the
bandwidth audit's pub and signs, and the window salt, positions and operands' roots a seed reads), with no case per scheme;
the decoder's refusals each name their code; an output held to a supplied constant is checked once; a block's part of a
supplied input keeps C's mark; and an evaluation that reads a commitment registered after its coin is refused by name."""
from __future__ import annotations

import pytest

from verity.circuits import supplied as S


def _entry(name, fn, args, layout, bits, n=1):
    return [name, n, {"supplied": fn, "args": args, "layout": layout, "bits": bits}]


#: (entry, its nodes as (binding, width, leaves)) at the deployed widths
SERVED = [
    (_entry("round_3", "pous.session.round_coin/v1", {"round": 3}, "halves", 96), [(S.SUPPLIED, 32, 3)]),
    (_entry("header", "pous.session.header/v1", {}, "le", 6048), [(S.SUPPLIED, 1, 6048)]),
    (_entry("context", "pous.session.context/v1", {}, "le", 848), [(S.SUPPLIED, 1, 848)]),
    (_entry("base_5", "pous.setup.base/v1", {"draw": 5, "L": 144}, "le", 1152), [(S.SUPPLIED, 16, 72)]),
    (_entry("slot_5", "pous.setup.slot/v1", {"draw": 5, "n": 2}, "le", 16), [(S.SUPPLIED, 16, 1)]),
    (_entry("pub_9", "poub.read.pub/v1", {"draw": 9, "K": 5, "masked": 1, "N": 256}, "le", 6368), [(S.SUPPLIED, 16, 398)]),
    (_entry("signs_9", "poub.read.signs/v1", {"draw": 9, "rows": 8, "bytes": 128}, "le", 8192), [(S.SUPPLIED, 16, 512)]),
    (_entry("salt", "window.salt/v1", {}, "halves", 512), [(S.SUPPLIED, 32, 16)]),
    (_entry("root_a", "commitment.root/v1", {"commitment": "a/0@1"}, "digest", 512), [(S.SUPPLIED, 1, 512)]),
    (_entry("position", "unit.position/v1", {"unit": 7}, "halves", 64), [(S.SUPPLIED, 16, 4)]),
    (_entry("one", "const.u16/v1", {"value": 1}, "le", 16), [(S.SUPPLIED, 16, 1)]),
]


@pytest.mark.parametrize("entry,nodes", SERVED, ids=[e[0] for e, _ in SERVED])
def test_each_row_a_verifier_makes_holds_its_mark(entry, nodes):
    assert S.check_entry(entry, nodes) == entry[2]


def test_unmarked_entries_are_unchanged():
    assert S.check_entry(["x", 2], [("run", 16, 4), ("run", 32, 1)]) is None
    assert S.check_entry(["w", 1], [("registration", 16, 8)]) is None


@pytest.mark.parametrize("entry,nodes,code", [
    (["k", 1, {"supplied": "window.salt/v1"}], [(S.SUPPLIED, 32, 16)], "E_SUPPLIED_MARK"),
    (["k", 1, {"supplied": "window.salt/v1", "args": {}, "layout": "bytes", "bits": 512}], [(S.SUPPLIED, 32, 16)],
     "E_SUPPLIED_MARK"),
    (_entry("k", "pous.session.round_coin/v1", {"round": 3}, "halves", 96), [("registration", 32, 3)], "E_INPUTS_MIXED"),
    (_entry("k", "pous.session.round_coin/v1", {"round": 3}, "halves", 96), [(S.SUPPLIED, 32, 1), ("run", 32, 2)],
     "E_INPUTS_MIXED"),
    (["k", 1], [(S.SUPPLIED, 32, 3)], "E_INPUTS_MIXED"),
    (_entry("k", "pous.session.round_key/v1", {"round": 3}, "halves", 64), [(S.SUPPLIED, 32, 2)], "E_SUPPLIED_FN"),
    (_entry("k", "pous.session.round_coin/v1", {}, "halves", 96), [(S.SUPPLIED, 32, 3)], "E_SUPPLIED_ARGS"),
    (_entry("k", "const.u16/v1", {"value": 1 << 16}, "le", 16), [(S.SUPPLIED, 16, 1)], "E_SUPPLIED_ARGS"),
    (_entry("k", "pous.setup.base/v1", {"draw": 5, "L": "x"}, "le", 1152), [(S.SUPPLIED, 16, 72)], "E_SUPPLIED_ARGS"),
    (_entry("k", "pous.session.round_coin/v1", {"round": 3}, "halves", 64), [(S.SUPPLIED, 32, 2)], "E_SUPPLIED_WIDTH"),
    (_entry("k", "pous.session.round_coin/v1", {"round": 3}, "halves", 96), [(S.SUPPLIED, 32, 2), (S.SUPPLIED, 16, 2)],
     "E_SUPPLIED_WIDTH"),
    (_entry("k", "pous.session.context/v1", {}, "halves", 848), [(S.SUPPLIED, 16, 53)], "E_SUPPLIED_WIDTH"),
    (_entry("k", "commitment.root/v1", {"commitment": "a"}, "digest", 512), [(S.SUPPLIED, 32, 16)], "E_SUPPLIED_WIDTH"),
])
def test_refusals(entry, nodes, code):
    with pytest.raises(S.SuppliedError, match=f"^{code}: ") as e:
        S.check_entry(entry, nodes)
    assert e.value.code == code


def test_an_output_held_to_a_supplied_constant():
    """Every unit's ok word (a 16-bit output leaf each, 4 units) held to the one leaf of `const.u16/v1` at 1, as an
    opening's verdict and the storage setup's ok are."""
    one = S.check_entry(*SERVED[-1])
    inputs = {"one": (one, 16, 1), "base_5": (SERVED[3][0][2], 16, 72)}
    S.check_held([10, 14, {"equals": ["one", 0, 1]}], 16, inputs)
    for bad, width in ([[10, 14, {"equals": ["base_6", 0, 1]}], 16], [[10, 14, {"equals": ["one", 0, 1]}], 1],
                       [[10, 14, {"equals": ["one", 0, 3]}], 16], [[10, 14, {"equals": ["one", 1, 1]}], 16],
                       [[10, 14, {"interval": [0, 1]}], 16]):
        with pytest.raises(S.SuppliedError, match="^E_HELD: "):
            S.check_held(bad, width, inputs)


def test_a_blocks_part_names_cs_input_and_range():
    """The opening's key reads leaf 2 of round 3's coin (i_j's low half): a whole leaf, so no second function gives i_j."""
    mark = S.check_entry(*SERVED[0])
    assert S.part("round_3", mark, 2, 3, 32) == {**mark, "bits": 32, "of": ["round_3", 2, 3]}
    with pytest.raises(S.SuppliedError, match="^E_SUPPLIED_WIDTH: "):
        S.part("round_3", mark, 2, 4, 32)


def test_the_order_rule():
    S.check_order({"store": 0, "session": 1}, coin=2)
    with pytest.raises(S.SuppliedError, match=r"^E_SUPPLIED_ORDER: \['late'\] registered after the coin"):
        S.check_order({"store": 0, "late": 2}, coin=2)


def test_every_function_reads_only_public_things():
    for f in S.FUNCTIONS.values():
        assert f.origin in S.ORIGINS and set(f.reads) <= set(S.READS) and f.reads
        assert (f.origin == "coin") == (f.reads == ("coin",)), f.name
