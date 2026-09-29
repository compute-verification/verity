"""The closure-draw plan (`verity_pouw.circuit.plan`): closures, the law seam, openings and the profile."""
from __future__ import annotations

from fractions import Fraction

import pytest

from verity.proofs.profile import Stratified, Stratum
from verity.randomness import derive

from verity_pouw.circuit import ncp2
from verity_pouw.circuit import plan as PL


def _key(tag: bytes = b"r"):
    return derive(b"verifier-secret", PL.DRAW_DOMAIN, {"receipt": tag})


def test_a_tiles_closure_is_its_strips_digest_path_and_key():
    lay = PL.layout(ncp2.Shape(48, 64, 32))
    us = lay.units
    for tile, closure in lay.closure.items():
        rs, cs = us[tile].where
        roles = [us[u].role for u in closure]
        assert roles[:3] == ["tile", "form_x", "digest"] and roles[-1] == "key"
        assert us[closure[1]].where == (rs,) and us[closure[2]].where == (rs,)
        assert lay.y_strip[tile] == cs                     # its Y unit is the weight Program's, column strip cs
        # the unbalanced tree: strips 0 and 1 sit under a node below the root, strip 2 under the root alone
        assert len([r for r in roles if r == "node"]) == (2 if rs < 2 else 1)


def _by_role(lay):
    return {s.role: s for s in lay.strata}


def test_the_strata_are_the_verifiers_own():
    """One stratum per template, its units as merged ranges, in the order of their first unit (`Flock/Draw.lean` `strataOf`
    over core's evaluation of the partition query), covering the population exactly."""
    from verity.ir import partition_object as PO
    from verity_pouw.circuit import partition as P

    shape = ncp2.Shape(32, 64, 32)
    lay = PL.layout(shape)
    want: dict[str, list[list[int]]] = {}
    for _k, tid, lo, n in PO.evaluate(P.program(shape), P.query(shape)).ranges:
        rs = want.setdefault(tid, [])
        if rs and rs[-1][1] == lo:
            rs[-1][1] = lo + n
        else:
            rs.append([lo, lo + n])
    assert [(s.name, s.ranges()) for s in lay.strata] == list(want.items())
    assert sorted(u for s in lay.strata for u in s.units) == list(range(len(lay.units)))
    assert {s.role for s in lay.strata} == {"digest", "node", "key", "form_x", "tile", "dequant"}   # Y is the weight's


def test_the_count_law_is_mains_stratified_rule():
    lay = PL.layout(ncp2.Shape(32, 64, 32))
    sizes = PL.CountLaw(8).sizes(lay.strata)
    n = len(lay.units)                                     # 2 digests, 1 node, the key, 2 X strips, 4 tiles, 1,024 outputs
    assert n == 1034
    assert sizes == {s.name: 8 if s.role == "dequant" else 1 for s in lay.strata}   # round(8·1024/1036); the rest at one


def test_the_work_law_sizes_tiles_by_work_and_the_rest_at_its_floor():
    lay = PL.layout(ncp2.Shape(32, 64, 32))
    tile = _by_role(lay)["tile"]
    assert PL.WorkLaw(3).sizes(lay.strata) == {s.name: 3 if s is tile else 1 for s in lay.strata}   # ⌈3·W/W⌉, floor of one
    assert PL.WorkLaw(9).sizes(lay.strata) == {s.name: 4 if s is tile else 1 for s in lay.strata}   # capped at 4 tiles


def test_the_work_law_is_roots_rule():
    """#362's example: T0 has 3 units of work 3, T1 3 units of work 1, T2 4 units of work 0; `work:2` draws 2, 1 and 1."""
    strata = [PL.StratumSpec("T0", "tile", (0, 1, 2), 9), PL.StratumSpec("T1", "tile", (3, 4, 5), 3),
              PL.StratumSpec("T2", "dequant", (6, 7, 8, 9), 0)]
    assert PL.WorkLaw(2).sizes(strata) == {"T0": 2, "T1": 1, "T2": 1}


def test_the_record_budget_is_27713():
    """(1 − 0.001)^K ≤ 2⁻⁴⁰ first at K = 27,713 (#362's `record_sizing`); 27,712 misses (`record_sizing_one_fewer`)."""
    assert PL.record_k(Fraction(1, 1000), Fraction(1, 2 ** 40)) == PL.RECORD_K == 27_713
    assert 999 ** 27_712 * 2 ** 40 > 1000 ** 27_712
    assert PL.record_k(Fraction(1, 1000), Fraction(1, 100)) == 4_603


def test_a_plan_proves_each_drawn_tile_with_its_closure_once():
    lay = PL.layout(ncp2.Shape(32, 64, 32))
    plan = PL.draw(_key(), lay, PL.WorkLaw(2))
    assert plan == PL.draw(_key(), lay, PL.WorkLaw(2))                    # the verifier's key fixes it
    tiles = [u for u in plan.drawn if lay.units[u].role == "tile"]
    want = set(plan.drawn).union(*(lay.closure[t] for t in tiles))
    assert plan.proved_y == tuple(sorted({lay.y_strip[t] for t in tiles}))
    assert set(plan.proved) == want and len(plan.proved) == len(want)
    assert len(tiles) == 2 and len(plan.drawn) == 2 + len(lay.strata) - 1   # each other stratum at its floor of one


def test_the_work_table_and_law_object_are_362s():
    """`work_table` is the verifier's `--work-table` (W_ref per tile, 0 elsewhere, and each template's floor f_s);
    `law_object` is the law `flock-verify draw --work K` derives from it (`Draw.deriveWork`, `Law.toJson`). A floor from the
    verifier's table raises a zero-work stratum's draws to it (X-SPC-81)."""
    lay = PL.layout(ncp2.Shape(32, 64, 32))
    tile, deq = _by_role(lay)["tile"], _by_role(lay)["dequant"]
    table = PL.work_table(lay)
    assert table == {s.name: {"work": lay.tile_work() if s is tile else 0, "floor": 1} for s in lay.strata}
    law = PL.law_object(lay, PL.WorkLaw(3))
    assert law == {"law": "work", "k": 3, "strata": [{"template": s.name, "units": s.ranges(), "work": table[s.name]["work"],
                                                      "floor": 1, "k": 3 if s is tile else 1} for s in lay.strata]}
    floored = PL.WorkLaw(3, floors=((deq.name, 5),))
    assert PL.work_table(lay, floored)[deq.name] == {"work": 0, "floor": 5}
    assert {s["template"]: (s["floor"], s["k"]) for s in PL.law_object(lay, floored)["strata"]}[deq.name] == (5, 5)


def test_openings_are_one_leaf_per_row():
    """A and the weight are committed a leaf per row (X-SPC-57): a proved strip opens exactly its 16 rows."""
    lay = PL.layout(ncp2.Shape(32, 64, 32))
    by_role = {u.role: u for u in lay.units if u.role == "form_x"}
    assert PL.openings(lay, [by_role["form_x"].index]) == tuple(("a", r) for r in range(16 * by_role["form_x"].where[0],
                                                                                         16 * by_role["form_x"].where[0] + 16))
    assert PL.openings(lay, [], [1]) == (*(("b", r) for r in range(16, 32)), ("y", 1))   # the weight Program's Y unit 1,
    # and the call's Y input at its strip beside that unit's output (X-SPC-95)


def test_the_profile_is_stratified_over_the_tile_strata():
    lay = PL.layout(ncp2.Shape(32, 64, 32))
    tile = _by_role(lay)["tile"]
    assert PL.profile(lay, {tile.name: 1}) == Stratified((Stratum(tile.name, 4, 1),))
    # one draw of four tiles at δ = 1/4: three unsound tiles escape with probability exactly 1/4
    bound = PL.unsound_work_bound(lay, {tile.name: 1}, 0.25)
    assert 3 * lay.tile_work() <= bound < 3.01 * lay.tile_work()


def test_traced_as_is_the_compositions_entry_point():
    """What #311's PoUW adapter calls: the circuit for `ncp-v2`, nothing for a scheme without one."""
    import pytest

    from verity_pouw.circuit import partition as P
    from verity_pouw.circuit import traced_as
    from verity_pouw.protocol import DomainError

    t = traced_as("ncp-v2-shift24", 32, 64, 32)
    assert t.definition is ncp2.linear(ncp2.Shape(32, 64, 32)) and t.query == P.query(ncp2.Shape(32, 64, 32))
    assert t.served == 32 * 32 and t.layout().closure == PL.layout(ncp2.Shape(32, 64, 32)).closure
    assert traced_as("ncp-v1", 32, 64, 32) is None and traced_as("pearl-fp8-v4", 32, 64, 32) is None
    with pytest.raises(DomainError):
        traced_as("ncp-v2", 30, 64, 32)
    assert traced_as("ncp-v2", 32, 64, 32).definition is ncp2.linear(ncp2.Shape(32, 64, 32), "word")   # the scheme pins P
    assert t.weight is ncp2.weight(ncp2.Shape(32, 64, 32)) and t.kind == "shift24"


def test_the_default_work_law_draws_a_calls_share_of_the_windows_budget():
    """K = 27,713 is a window's budget (X-SPC-83): alone, a call is capped at its units instead of raising; in a window each
    call draws ⌈K·W_c/W_window⌉, so C equal calls draw about K between them, not C·K."""
    lay = PL.layout(ncp2.Shape(32, 64, 32))
    tile = _by_role(lay)["tile"]
    assert PL.WorkLaw().sizes(lay.strata) == {s.name: 4 if s is tile else 1 for s in lay.strata}   # K capped at its units
    laws = PL.window_laws([lay] * 10_000)
    assert laws[0].budget(lay.strata) == -(-PL.RECORD_K // 10_000) == 3
    assert laws[0].sizes(lay.strata) == {s.name: 3 if s is tile else 1 for s in lay.strata}
    assert PL.law_object(lay, laws[0])["k"] == 3                     # #362's law at the call's share
    assert sum(law.sizes(lay.strata)[tile.name] for law in laws) <= PL.RECORD_K + 10_000 * len(lay.strata)


def test_a_tiles_closure_contains_the_tile_itself():
    """A drawn tile is proved itself, not only what it reads: `t ∈ cl t`, as Lean's `Layout.cl` inserts the tile's own unit.
    Without it a wrong tile over right strips would count as zero harm (the Flock red team's C1 on #390)."""
    for shape in (ncp2.Shape(32, 64, 32), ncp2.Shape(48, 64, 32)):
        lay = PL.layout(shape)
        tiles = {u.index for u in lay.units if u.role == "tile"}
        assert set(lay.closure) == tiles
        assert all(lay.closure[t][0] == t for t in tiles)
        plan = PL.draw(_key(), lay, PL.WorkLaw(len(tiles)))                 # every tile drawn
        assert tiles <= set(plan.drawn) <= set(plan.proved)
        assert all(set(lay.closure[t]) <= set(plan.proved) for t in tiles)


def test_a_prover_s_layout_mapping_a_tile_to_its_x_strip_is_refused():
    """The Layout theorems hold only for the window's own layout, so the verifier derives each tile's own unit and its
    closure from the shape (`Traced.layout`) and never takes a prover's. A layout mapping a tile to its X strip, C1 again,
    is refused, as a closure headed by the X strip or as the X strip in the tile stratum: by the verifier's comparison
    (`Traced.check_layout`) and by `draw`'s own check."""
    from dataclasses import replace

    from verity_pouw.circuit import traced_as
    from verity_pouw.protocol import DomainError

    t = traced_as("ncp-v2-shift24", 32, 64, 32)
    lay = t.layout()
    t.check_layout(lay)
    tile = next(u.index for u in lay.units if u.role == "tile")
    xs = lay.closure[tile][1]
    assert lay.units[xs].role == "form_x"
    headed_by_x = replace(lay, closure={**lay.closure, tile: (xs, *(u for u in lay.closure[tile] if u not in (tile, xs)))})
    strata = tuple(replace(s, units=tuple(xs if u == tile else u for u in s.units)) if s.role == "tile" else s
                   for s in lay.strata)
    x_as_tile = replace(lay, strata=strata, closure={(xs if u == tile else u): c for u, c in lay.closure.items()})
    for bad in (headed_by_x, x_as_tile):
        with pytest.raises(DomainError):
            t.check_layout(bad)
        with pytest.raises(DomainError):
            PL.draw(_key(), bad, PL.WorkLaw())


def _anchors(call, **kw):
    from verity_pouw.circuit.anchors import Anchors

    f = {"forward": call.forward, "layer": call.layer, "site": call.site, **kw}
    return Anchors(call.salt, f["forward"], f["layer"], f["site"], call.weight, tuple(call.rowids), tuple(call.colids))


def test_equal_calls_under_one_key_draw_different_tiles():
    """Each stratum's subset is keyed by the call's index, (forward, position), with its template (X-SPC-96): two equal
    calls of a window, under the one key, draw different tiles, and one call draws the same tiles every time. The strata
    stay one per template of the call's partition (X-SPC-84); only the randomness is per call."""
    import circuit_fixtures as F
    from verity_pouw.circuit import traced_as

    call = F.call(64, 64, 64)
    t, law = traced_as("ncp-v2-shift24", 64, 64, 64), PL.WorkLaw(4)
    tiles = lambda plan: [u for u in plan.drawn if t.layout().units[u].role == "tile"]  # noqa: E731
    first = t.draw(_key(), law, _anchors(call))
    assert tiles(first) == tiles(t.draw(_key(), law, _anchors(call)))
    others = [tiles(t.draw(_key(), law, _anchors(call, **kw))) for kw in ({"forward": 4}, {"site": 3}, {"layer": 6})]
    assert all(o != tiles(first) for o in others)
    assert [s.name for s in t.layout().strata] == list(dict.fromkeys(u.template for u in t.layout().units))


def test_the_verifier_draws_over_its_own_layout():
    """`Traced.draw(key, law, anchors)` takes no layout: it is `plan.draw` over `Traced.layout()` (X-SPC-97)."""
    import inspect

    import circuit_fixtures as F
    from verity_pouw.circuit import Traced, traced_as

    call = F.call(32, 64, 32)
    t = traced_as("ncp-v2-shift24", 32, 64, 32)
    assert list(inspect.signature(Traced.draw).parameters) == ["self", "key", "law", "anchors"]
    an = _anchors(call)
    assert t.draw(_key(), PL.WorkLaw(2), an) == PL.draw(_key(), t.layout(), PL.WorkLaw(2), an.call_index)


def test_the_call_s_y_is_linked_to_the_weight_program():
    """The call reads its weight's Y strips and c₀ as inputs, so the plan opens them at each drawn tile's column strip
    beside the weight Program's outputs there, and the verifier compares the two (`Traced.check_link`, X-SPC-95). Another
    weight's Y, or one c₀ changed, is refused."""
    import pytest

    import circuit_fixtures as F
    from verity.evaluation import evaluate
    from verity_pouw.circuit import ncp2, traced_as
    from verity_pouw.protocol import DomainError

    call = F.call(32, 64, 32)
    t = traced_as("ncp-v2-shift24", 32, 64, 32)
    plan = t.draw(_key(), PL.WorkLaw(2), _anchors(call))
    assert {cs for role, cs in plan.openings if role == "y"} == set(plan.proved_y) != set()
    weight_out = list(evaluate(ncp2.weight(call.shape), *call.weight_args()))
    args = call.args()
    t.check_link(plan, args, weight_out)
    other = F.call(32, 64, 32, seed=8)                                        # another weight B′, with its own Y and c₀
    swapped = list(args)
    swapped[2:4] = other.args()[2:4]
    with pytest.raises(DomainError):
        t.check_link(plan, swapped, weight_out)
    cs = plan.proved_y[0]
    bumped = list(args)
    bumped[3] = [v + 1 if 16 * cs <= i < 16 * cs + 16 else v for i, v in enumerate(args[3])]
    with pytest.raises(DomainError):
        t.check_link(plan, bumped, weight_out)
