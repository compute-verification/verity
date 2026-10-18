"""The window's static Program (`pouw.program`): the layout the padded-credit rule covers passes the shape check under
both hashings and both B plans, main's window is refused, each refusal fires on the one change that breaks its clause,
the tally's limb Program is exact and refuses a wrong limb, and the cost of each B plan at W5's window."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from verity.core.protocols.pouw import program as Pg
from verity.core.protocols.pouw import window as Wd

K = 4096
SHAPE = Wd.shape_of([{"m": 16, "n": 3, "tile0": 0}, {"m": 16, "n": 2, "tile0": 5}])


def _kinds(lay: Pg.Layout) -> set[str]:
    return {p.split(":")[1].strip() for p in Pg.problems(lay)}


def _pos(lay: Pg.Layout, name: str, **kw) -> Pg.Layout:
    return replace(lay, positions={**lay.positions, name: replace(lay.positions[name], **kw)})


def _unit(lay: Pg.Layout, name: str, **kw) -> Pg.Layout:
    return replace(lay, units={**lay.units, name: replace(lay.units[name], **kw)})


@pytest.mark.parametrize("hashing", Pg.HASHINGS)
@pytest.mark.parametrize("plan", Pg.B_PLANS)
def test_the_rules_layout_passes_the_shape_check(hashing, plan):
    lay = Pg.layout(SHAPE, K, hashing=hashing, b_plan=plan)
    assert Pg.problems(lay) == []
    tiles = [u for u in lay.units.values() if u.kind == "tile"]
    assert len(tiles) == Wd.shape_tiles(SHAPE)
    pieces = 2 + K // Pg.rowk.ROW_K
    assert sum(u.kind == "form" for u in lay.units.values()) == pieces * sum(c.m + c.n for c in SHAPE)
    head = lay.units["a0.0/head"]
    assert (head.definition == Pg.SEEDED_HEAD) == (hashing == "h3") and (head.anchors == ()) == (hashing == "h3")


def test_the_layout_is_the_shapes_alone():
    a, b = Pg.layout(SHAPE, K), Pg.layout(Wd.shape_of([c.to_json() for c in SHAPE]), K)
    assert a == b
    other = Pg.layout(Wd.shape_of([{"m": 32, "n": 3, "tile0": 0}, {"m": 16, "n": 2, "tile0": 5}]), K)
    assert len(other.units) > len(a.units)


def test_mains_window_is_refused():
    kinds = _kinds(Pg.current(SHAPE, K))
    assert {"unsourced-gate", "untallied-gate", "operand-after-certificate", "stale-b-rows", "filler"} <= kinds


@pytest.mark.parametrize("kind, change", [
    ("late-selection", lambda l: _pos(l, "a0.1.ok", phase="forming")),
    ("late-selection", lambda l: _pos(l, "b1.0.ok", phase="epoch", registered=True)),
    ("operand-after-certificate", lambda l: _pos(l, "a0.2.p", writer=None)),
    ("operand-after-certificate", lambda l: _pos(l, "a0.2.ok", certifies="a0.3.x")),
    ("unsourced-gate", lambda l: _pos(l, "a1.0.ok", writer=None)),
    ("unsourced-gate", lambda l: _unit(l, "a1.0/admit", plan="drawn")),
    ("untallied-gate", lambda l: _unit(l, "sum0", reads=l.units["sum0"].reads[2:])),
    ("missing-proved", lambda l: _unit(l, "a0.0/ok", plan="drawn")),
    ("missing-proved", lambda l: _unit(l, "prod1", plan="weighted")),
    ("missing-proved", lambda l: _unit(l, "b0.1/tail", plan="drawn")),
    ("filler", lambda l: _pos(l, "filler.a.ok", constant=1)),
    ("filler", lambda l: _pos(l, "filler.b.p", constant=None)),
    ("advice", lambda l: _unit(l, "add1", advice=("carry",))),
    ("tally-reads", lambda l: _unit(l, "sum1", reads=l.units["sum1"].reads + ("a1.0.x",))),
    ("stale-b-rows", lambda l: _pos(l, "b0.2.p", tree="window-b")),
    ("unknown-definition", lambda l: _unit(l, "t0.0.0", definition="Pc8TileCheap")),
    ("unproved-noise", lambda l: _unit(l, "b0.0/head", anchors=(("e", "verity/pouw/anchor/pearl-c/eb/v0"),))),
    ("unnoised", lambda l: _unit(l, "b1.1/head", anchors=())),
    ("draw", lambda l: replace(l, draw="by-ok")),
])
def test_each_refusal(kind, change):
    lay = change(Pg.layout(SHAPE, K))
    assert kind in _kinds(lay)


def test_a_forming_that_reads_a_stray_value_is_refused():
    lay = Pg.layout(SHAPE, K)
    lay = replace(lay, positions={**lay.positions, "a0.4.stats": Pg.Pos("activation")})
    lay = _unit(lay, "a0.4/form1", reads=lay.units["a0.4/form1"].reads + ("a0.4.stats",))
    assert "unclosed-forming" in _kinds(lay)


def test_ea_is_no_anchor_under_h3():
    lay = Pg.layout(SHAPE, K, hashing="h3")
    lay = _unit(lay, "a0.0/head", definition=Pg.rowk.Pc8RowAHeadDef.name,
                anchors=(("e", Pg.rowk.DERIVATIONS8["ea"].rule),))
    assert "anchored-ea" in _kinds(lay)
    assert Pg.problems(Pg.layout(SHAPE, K, hashing="h0")) == []


def test_a_weighted_forming_draws_at_its_static_cone():
    lay = Pg.layout(SHAPE, K, b_plan="weighted")
    assert {u.fan_out for n, u in lay.units.items() if n.startswith("b0.")} == {SHAPE[0].m + 1}
    assert "fan-out" in _kinds(_unit(lay, "b0.0/tail", fan_out=SHAPE[0].m + 2))


def _calls():
    return [{"a": [(1, 1)] * 10 + [(0, 0)] * 6, "sb": 3}, {"a": [(1, 1)] * 15 + [(0, 1)], "sb": 2}]


def test_the_tally_is_the_masked_sum_in_limbs():
    units = Pg.tally(_calls())
    assert Pg.tally_problem(units, 2) is None
    real = 10 * 3 + 15 * 2
    assert units[-1]["writes"] == [Wd.level(Wd.floor_point(real)), 1]
    assert all(0 <= w < 1 << 16 for u in units for w in list(u["reads"]) + list(u["writes"]))
    big = Pg.tally([{"a": [(1, 1)] * (1 << 13), "sb": (1 << 17) - 1}] * 3)
    assert Pg.tally_problem(big, 3) is None
    assert Pg._value(big[-1]["reads"]) == 3 * (1 << 13) * ((1 << 17) - 1)


def test_the_tally_reads_each_gate_through_one():
    forged = [{"a": [(2, 1)] * 16, "sb": 3}]
    assert Pg.tally(forged)[0]["writes"] == [0]
    assert Pg.tally(forged)[-1]["writes"] == [0, 0]


@pytest.mark.parametrize("unit, field, value", [(1, "writes", [31, 0]), (2, "writes", [61, 0, 0]),
                                                (4, "writes", [60, 1, 0]), (5, "reads", [30, 0, 0, 31, 0]),
                                                (6, "writes", [9, 1])])
def test_a_wrong_limb_is_refused(unit, field, value):
    units = Pg.tally(_calls())
    units[unit] = {**units[unit], field: value}
    assert Pg.tally_problem(units, 2) is not None


def test_the_cost_of_each_b_plan_at_w5s_window():
    w5 = Wd.shape_of([{"m": 16, "n": 4096, "tile0": 0}, {"m": 16, "n": 14336, "tile0": 64}])
    c = Pg.cost(w5, K, Fraction(1, 100), Fraction(1, 2 ** 40))
    assert (c["n"], c["k_tiles"], c["pieces_per_row"], c["rows_b"]) == (294913, 2759, 4, 18432)
    assert c["proved"]["b_pieces_per_salt"] == 73728
    assert c["weighted"]["draws_per_window"] == 5705
    assert 2900 < c["weighted"]["b_formings_per_window"] < 2950
    assert c["window_fixed"] == {"ok": 32, "admit": 32, "a_pieces": 128, "tally": 7}
