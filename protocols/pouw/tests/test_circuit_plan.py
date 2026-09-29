"""The closure-draw plan (`verity_pouw.circuit.plan`): closures, the law seam, openings and the profile."""
from __future__ import annotations

from fractions import Fraction

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
        assert roles[:4] == ["tile", "form_x", "form_y", "digest"] and roles[-1] == "key"
        assert us[closure[1]].where == (rs,) and us[closure[2]].where == (cs,) and us[closure[3]].where == (rs,)
        # the unbalanced tree: strips 0 and 1 sit under a node below the root, strip 2 under the root alone
        assert len([r for r in roles if r == "node"]) == (2 if rs < 2 else 1)


def test_the_count_law_is_mains_stratified_rule():
    lay = PL.layout(ncp2.Shape(32, 64, 32))
    sizes = PL.CountLaw(8).sizes(lay.strata)
    tile, rest = lay.strata
    assert sizes == {tile.name: 1, rest.name: 8}          # max(1, round(8·4/1028)), round(8·1024/1028)


def test_the_work_law_sizes_tiles_by_work_and_the_rest_at_its_floor():
    lay = PL.layout(ncp2.Shape(32, 64, 32))
    tile, rest = lay.strata
    assert PL.WorkLaw(3).sizes(lay.strata) == {tile.name: 3, rest.name: 1}   # ⌈3·W/W⌉, and the floor of one
    assert PL.WorkLaw(9).sizes(lay.strata) == {tile.name: 4, rest.name: 1}   # capped at the stratum's 4 tiles


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
    plan = PL.draw(_key(), lay, PL.CountLaw(8))
    assert plan == PL.draw(_key(), lay, PL.CountLaw(8))                  # the verifier's key fixes it
    tiles = [u for u in plan.drawn if lay.units[u].role == "tile"]
    want = set(plan.drawn).union(*(lay.closure[t] for t in tiles))
    assert set(plan.proved) == want and len(plan.proved) == len(want)
    assert all(lay.units[u].role in ("tile", "dequant") for u in plan.drawn)


def test_openings_are_one_leaf_per_row():
    """A and the weight are committed a leaf per row (X-SPC-57): a proved strip opens exactly its 16 rows."""
    lay = PL.layout(ncp2.Shape(32, 64, 32))
    by_role = {u.role: u for u in lay.units if u.role in ("form_x", "form_y")}
    assert PL.openings(lay, [by_role["form_x"].index]) == tuple(("a", r) for r in range(16 * by_role["form_x"].where[0],
                                                                                         16 * by_role["form_x"].where[0] + 16))
    assert PL.openings(lay, [by_role["form_y"].index]) == tuple(("b", r) for r in range(16 * by_role["form_y"].where[0],
                                                                                         16 * by_role["form_y"].where[0] + 16))


def test_the_profile_is_stratified_over_the_tile_strata():
    lay = PL.layout(ncp2.Shape(32, 64, 32))
    tile = lay.strata[0]
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

    t = traced_as("ncp-v2", 32, 64, 32)
    assert t.definition is ncp2.linear(ncp2.Shape(32, 64, 32)) and t.query == P.query(ncp2.Shape(32, 64, 32))
    assert t.served == 32 * 32 and t.layout().closure == PL.layout(ncp2.Shape(32, 64, 32)).closure
    assert traced_as("ncp-v1", 32, 64, 32) is None and traced_as("pearl-fp8-v4", 32, 64, 32) is None
    with pytest.raises(DomainError):
        traced_as("ncp-v2", 30, 64, 32)
