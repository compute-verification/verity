"""The closure-draw plan (`verity.core.protocols.pouw.circuit.plan`): closures, the law seam, openings and the profile."""
from __future__ import annotations

from fractions import Fraction

import pytest

from verity.core.protocols.profile import Stratified, Stratum
from verity.ml.randomness import derive

from verity.core.protocols.pouw.circuit import ncp2
from verity.core.protocols.pouw.circuit import plan as PL


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
    from verity.ml.circuits import partition_object as PO
    from verity.core.protocols.pouw.circuit import partition as P

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
    """#362's rule, k_s = min(n_s, max(f_s, ⌈K·W_s/W⌉)): T0 has 3 units of work 3 and T2 4 units of work 0, and `work:2`
    draws 2 and 1; a floor of 3 on T2 draws 3. #362's own example also gives work to a second template (T1, 3 units of
    work 1), which a call's table may not (`hone`, `check_work_strata`), so the verifier refuses it."""
    import pytest

    t0, t1, t2 = (PL.StratumSpec("T0", "tile", (0, 1, 2), 9), PL.StratumSpec("T1", "tile", (3, 4, 5), 3),
                  PL.StratumSpec("T2", "dequant", (6, 7, 8, 9), 0))
    assert PL.WorkLaw(2).sizes([t0, t2]) == {"T0": 2, "T2": 1}
    assert PL.WorkLaw(2, floors=(("T2", 3),)).sizes([t0, t2]) == {"T0": 2, "T2": 3}
    with pytest.raises(ValueError, match="one stratum"):
        PL.WorkLaw(2).sizes([t0, t1, t2])


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

    from verity.core.protocols.pouw.circuit import partition as P
    from verity.core.protocols.pouw.circuit import traced_as
    from verity.core.protocols.pouw.protocol import DomainError

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
    deq = _by_role(lay)["dequant"]                                   # y's floor ⌈K_y·1,024/(10,000·1,024)⌉ (option B)
    assert laws[0].sizes(lay.strata) == {s.name: 3 if s in (tile, deq) else 1 for s in lay.strata}
    assert PL.law_object(lay, laws[0])["k"] == 3                     # #362's law at the call's share
    assert sum(law.sizes(lay.strata)[tile.name] for law in laws) <= PL.RECORD_K + 10_000 * len(lay.strata)



def test_y_is_drawn_at_root_s_option_b_floors_over_the_window():
    """Root's option B on X-SPC-105: each call's dequantization stratum has floor f_c = ⌈K_y·n_c/N⌉ in its own work table, N
    the window's y cells and K_y = K = 27,713, so k_c = min(n_c, f_c) meets k_c·N ≥ K_y·n_c or proves the stratum, and the
    window draws at most K_y + C y cells. The other zero-work templates keep floor 1, and the tiles' K_c don't move. A call
    alone proves all its y cells up to K_y. Over the window the escape of b_c wrong cells in each call is at most
    (1 − ∑b_c/N)^K_y (weighted AM–GM, root's §1), the bound the window pin states."""
    import random

    shapes = [ncp2.Shape(32, 64, 32)] * 3 + [ncp2.Shape(64, 64, 64)] * 2 + [ncp2.Shape(48, 64, 32)]
    lays = [PL.layout(s) for s in shapes] * 400
    laws = PL.window_laws(lays)
    n_y = [len(_by_role(lay)["dequant"].units) for lay in lays]
    big_n = sum(n_y)
    ks = []
    tiled = PL.WorkLaw(PL.RECORD_K, window_work=sum(lay.work() for lay in lays))     # the tiles' K_c, without y's floors
    for lay, law, n_c in zip(lays, laws, n_y):
        deq, table = _by_role(lay)["dequant"], PL.work_table(lay, law)
        f_c = -(-PL.RECORD_K * n_c // big_n)
        assert table[deq.name] == {"work": 0, "floor": f_c}
        assert all(table[s.name]["floor"] == 1 for s in lay.strata if s.role not in ("dequant",))
        k_c = law.sizes(lay.strata)[deq.name]
        assert k_c == min(n_c, f_c) and (k_c * big_n >= PL.RECORD_K * n_c or k_c == n_c)
        tile = _by_role(lay)["tile"].name
        assert law.sizes(lay.strata)[tile] == tiled.sizes(lay.strata)[tile]
        assert PL.law_object(lay, law)["strata"][[s.name for s in lay.strata].index(deq.name)]["floor"] == f_c
        ks.append(k_c)
    assert PL.RECORD_K <= sum(ks) <= PL.RECORD_K + len(lays)
    rng = random.Random(105)
    for _ in range(20):
        bad = [rng.randrange(0, n_c // 50 + 1) for n_c in n_y]
        per_call = 1.0
        for b, n_c, k_c in zip(bad, n_y, ks):
            per_call *= ((n_c - b) / n_c) ** k_c
        assert per_call <= (1 - sum(bad) / big_n) ** PL.RECORD_K * (1 + 1e-9)
    lay = PL.layout(ncp2.Shape(32, 64, 32))
    (alone,) = PL.window_laws([lay])
    assert alone.sizes(lay.strata)[_by_role(lay)["dequant"].name] == 32 * 32


def test_a_work_table_with_a_calls_work_in_two_strata_is_refused():
    """The window pin's `covers_window` holds only when each call's work is in one stratum (`hone`): with the cap K_c ≤ N_c,
    root's counterexample (4 units at work 2 and 2 at work 5, K = 8) draws 3 of the first where covering needs 3.6. The
    verifier's own layouts give work to the tile template only (X-SPC-84), and `work_table`, `WorkLaw` and `window_laws`
    refuse any layout that doesn't."""
    import dataclasses

    import pytest

    lay = PL.layout(ncp2.Shape(32, 64, 32))
    assert [s.role for s in lay.strata if s.work] == ["tile"]
    PL.work_table(lay, PL.window_laws([lay] * 3)[0])
    deq = _by_role(lay)["dequant"]
    two = dataclasses.replace(lay, strata=tuple(dataclasses.replace(s, work=len(s.units)) if s is deq else s
                                                 for s in lay.strata))
    for refused in (lambda: PL.work_table(two), lambda: PL.WorkLaw().sizes(two.strata), lambda: PL.window_laws([lay, two]),
                    lambda: PL.law_object(two, PL.WorkLaw()), lambda: PL.draw(_key(), two, PL.WorkLaw())):
        with pytest.raises(ValueError, match="one stratum"):
            refused()
    a, b = PL.StratumSpec("a", "tile", (0, 1, 2, 3), 2 * 4), PL.StratumSpec("b", "tile", (4, 5), 5 * 2)
    assert -(-6 * a.work // (a.work + b.work)) == 3 and 3 * 18 < 8 * a.work       # 3 drawn, 8·2·4/18 ≈ 3.6 needed
    with pytest.raises(ValueError, match="one stratum"):
        PL.WorkLaw(8).sizes((a, b))
    tile = _by_role(lay)["tile"]                                                 # a call's tiles split over two templates
    split = dataclasses.replace(lay, strata=tuple(x for s in lay.strata for x in (
        (dataclasses.replace(s, units=s.units[:1]), dataclasses.replace(s, name=s.name + "'", units=s.units[1:], work=0))
        if s is tile else (s,))))
    with pytest.raises(ValueError, match="one template"):
        PL.work_table(split)


def test_the_red_team_s_capped_window_is_refused():
    """The red team's window (20260929T1622Z): call 0 has 1,000 units at work 2 and 10 at work 1,000, call 1 100,000 at
    work 0 and 1 at work 1, and K = 5,000. Call 0's share 5,000 is capped at its 1,010 units, so its work-2 stratum would
    draw 169 of 1,000 (834 uncapped), and ten wrong units there escape with 0.156 where the window bound claims
    (1 − 20/12,001)^5,000 ≈ 2.4·10⁻⁴. `window_laws` refuses the window: call 0 does work in two strata."""
    from math import comb

    import pytest

    shape = ncp2.Shape(16, 64, 16)
    call0 = PL.PlanUnits(shape, (), (PL.StratumSpec("w2", "form_x", tuple(range(1000)), 2 * 1000),
                                  PL.StratumSpec("t0", "tile", tuple(range(1000, 1010)), 1000 * 10)), {})
    call1 = PL.PlanUnits(shape, (), (PL.StratumSpec("y1", "dequant", tuple(range(100_000)), 0),
                                  PL.StratumSpec("t1", "tile", (100_000,), 1)), {})
    total, k = call0.work() + call1.work(), 5000
    share = -(-k * call0.work() // total)
    assert (total, share) == (12_001, 5000) and share > 1010                     # the cap binds
    assert (-(-1010 * 2000 // call0.work()), -(-k * 2000 // call0.work())) == (169, 834)
    escape = comb(990, 169) / comb(1000, 169)
    assert 0.15 < escape < 0.16 and (1 - 20 / total) ** k < 2.5e-4
    with pytest.raises(ValueError, match="one stratum"):
        PL.window_laws([call0, call1], k=k)
    PL.window_laws([call1], k=k)                                                 # one work stratum a call: admitted


def test_a_window_that_does_no_work_is_refused():
    """The window pin's `hW`, 0 < W: `window_laws` refuses a window whose calls do no work (every stratum at work 0), so
    it is never drawn or accepted; one call's work is enough."""
    import pytest

    shape = ncp2.Shape(16, 64, 16)
    idle = PL.PlanUnits(shape, (), (PL.StratumSpec("t", "tile", (0,), 0), PL.StratumSpec("y", "dequant", (1, 2), 0)), {})
    busy = PL.PlanUnits(shape, (), (PL.StratumSpec("t", "tile", (0,), 5), PL.StratumSpec("y", "dequant", (1, 2), 0)), {})
    for window in ([idle], [idle, idle]):
        with pytest.raises(ValueError, match="does some work"):
            PL.window_laws(window)
    with pytest.raises(ValueError, match="at least one call"):
        PL.window_laws([])
    assert len(PL.window_laws([busy])) == 1


def test_a_window_with_no_y_cells_is_refused():
    """The window pin's `hN`, 0 < N: `window_laws` refuses a window with no dequantization units, so its y term is never
    claimed over an empty population; the verifier's own layouts always have them."""
    import pytest

    shape = ncp2.Shape(16, 64, 16)
    dry = PL.PlanUnits(shape, (), (PL.StratumSpec("t", "tile", (0, 1), 10), PL.StratumSpec("k", "key", (2,), 0)), {})
    for window in ([dry], [dry, dry]):
        with pytest.raises(ValueError, match="has y cells"):
            PL.window_laws(window)
    lay = PL.layout(shape)
    assert lay.work() > 0 and any(s.role == "dequant" for s in lay.strata)
    assert len(PL.window_laws([lay, lay])) == 2

def test_the_window_law_is_the_window_pin_s():
    """#418's `Audit/Window.lean` definitions, recomputed on ℕ from their formulas on random small windows, against
    `window_laws`: `callK` = min(N_c, max(1, ⌈K·W_c/W⌉)) is `WorkLaw.budget`, `windowK` = workRule(K_c, w_s, n_s, W_c, f_s)
    is `WorkLaw.sizes`, and the pin's hypotheses hold: `hone` (each call's work in one stratum), `hfy` (y's floors, v = 1 on
    the dequantization strata) and so `Covers` of w at K and of v at K_y, which `audit_window_split_of_record` reads."""
    import random

    def ceil(a, b):                                           # Lean's (a + b - 1) / b on ℕ
        return (a + b - 1) // b

    def work_rule(k, work, size, total, floor):               # `Work.lean`'s workRule
        return min(size, max(floor, ceil(k * work * size, total)))

    rng, shape = random.Random(418), ncp2.Shape(16, 64, 16)
    for _ in range(300):
        calls, k = rng.randint(1, 6), rng.choice([1, 2, 3, 7, 30, 200])
        layouts = []
        for c in range(calls):
            spec = [("tile", "tile", rng.randint(1, 8), rng.randint(1, 50))]
            spec += [(f"z{i}", rng.choice(["digest", "key", "form_x"]), rng.randint(1, 5), 0) for i in range(rng.randint(0, 2))]
            spec += [("y", "dequant", rng.randint(1, 20), 0)]
            rng.shuffle(spec)
            at, strata = 0, []
            for name, role, size, w in spec:
                strata.append(PL.StratumSpec(name, role, tuple(range(at, at + size)), w * size))
                at += size
            layouts.append(PL.PlanUnits(shape, (), tuple(strata), {}))
        floors = tuple((z, rng.randint(1, 4)) for z in ("z0", "z1") if rng.random() < 0.5)
        laws = PL.window_laws(layouts, k=k, floors=floors)
        big_w = sum(s.work for lay in layouts for s in lay.strata)
        big_n = sum(len(s.units) for lay in layouts for s in lay.strata if s.role == "dequant")
        for lay, law in zip(layouts, laws):
            w_c, n_c = sum(s.work for s in lay.strata), sum(len(s.units) for s in lay.strata)
            k_c = min(n_c, max(1, ceil(k * w_c, big_w)))                                   # callK
            assert law.budget(lay.strata) == k_c
            sizes, table = law.sizes(lay.strata), PL.work_table(lay, law)
            for s in lay.strata:
                n_s, w_s, f_s, v_s = len(s.units), table[s.name]["work"], table[s.name]["floor"], int(s.role == "dequant")
                assert sizes[s.name] == work_rule(k_c, w_s, n_s, w_c, f_s)                  # windowK
                assert w_s == 0 or w_c == w_s * n_s                                         # hone
                assert k * v_s * n_s <= f_s * big_n                                         # hfy at K_y = K
                assert k * w_s * n_s <= sizes[s.name] * big_w or sizes[s.name] == n_s       # Covers w K
                assert k * v_s * n_s <= sizes[s.name] * big_n or sizes[s.name] == n_s       # Covers v K_y


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

    from verity.core.protocols.pouw.circuit import traced_as
    from verity.core.protocols.pouw.protocol import DomainError

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
    from verity.core.protocols.pouw.circuit.anchors import Anchors

    f = {"forward": call.forward, "layer": call.layer, "site": call.site, **kw}
    return Anchors(call.salt, f["forward"], f["layer"], f["site"], call.weight, tuple(call.rowids), tuple(call.colids))


def _window(t, anchors, source=bytes(range(32)), commitments=None):
    """A window over ``anchors`` (each of ``t``'s shape, with a commitment of its own unless ``commitments`` gives them):
    its `Receipt`, and the `AdmittedCalls` opened from it, with the fresh source replaced by ``source`` so the draws are
    reproducible."""
    import hashlib
    from unittest import mock

    from verity.core.protocols.pouw.circuit import anchors as A

    cs = commitments or [hashlib.sha512(repr(an).encode()).digest() for an in anchors]
    receipt = A.Receipt(tuple((an, t.shape, c) for an, c in zip(anchors, cs)))
    with mock.patch.object(A, "_fresh_source", lambda: source):
        return receipt, A.AdmittedCalls.open(receipt)


def test_equal_calls_under_one_key_draw_different_tiles():
    """Each stratum's subset is keyed by the call's index, (forward, position), with its template (X-SPC-96): equal calls of
    a window, under its one key, draw different tiles, and one call draws the same tiles under that key every time. The
    strata stay one per template of the call's partition (X-SPC-84); only the randomness is per call."""
    import circuit_fixtures as F
    from verity.core.protocols.pouw.circuit import traced_as

    call = F.call(64, 64, 64)
    t, law = traced_as("ncp-v2-shift24", 64, 64, 64), PL.WorkLaw(4)
    ans = [_anchors(call)] + [_anchors(call, **kw) for kw in ({"forward": 4}, {"site": 3}, {"layer": 6})]
    _receipt, window = _window(t, ans)
    tiles = lambda plan: [u for u in plan.drawn if t.layout().units[u].role == "tile"]  # noqa: E731
    drawn = [tiles(t.draw(law, an, window)) for an in ans]
    assert drawn[0] == tiles(PL.draw(window.key, t.layout(), law, ans[0].call_index))
    assert all(d != drawn[0] for d in drawn[1:])
    assert [s.name for s in t.layout().strata] == list(dict.fromkeys(u.template for u in t.layout().units))


def test_the_verifier_draws_over_its_own_layout():
    """`Traced.draw(law, anchors, ledger)` takes no layout and no key: it is `plan.draw` over `Traced.layout()` (X-SPC-97)
    under the ledger's key."""
    import inspect

    import circuit_fixtures as F
    from verity.core.protocols.pouw.circuit import Traced, traced_as

    call = F.call(32, 64, 32)
    t = traced_as("ncp-v2-shift24", 32, 64, 32)
    assert list(inspect.signature(Traced.draw).parameters) == ["self", "law", "anchors", "ledger"]
    an = _anchors(call)
    _receipt, window = _window(t, [an])
    assert t.draw(PL.WorkLaw(2), an, window) == PL.draw(window.key, t.layout(), PL.WorkLaw(2), an.call_index)


def test_a_repeated_call_index_is_refused_before_any_draw(monkeypatch):
    """The window pin takes a window's per-call draws as independent (M1), and each call's subsets are keyed by its index
    under the one window key. A receipt lists each call index once, and `Traced.draw` admits the call to the window's
    `AdmittedCalls` before it draws: a call index already drawn, even with another weight id, is refused before `plan.draw`
    runs. A draw without a ledger is refused."""
    import hashlib

    import pytest

    import circuit_fixtures as F
    from verity.core.protocols.pouw.circuit import plan as plan_module
    from verity.core.protocols.pouw.circuit import traced_as
    from verity.core.protocols.pouw.circuit.anchors import Anchors, Receipt
    from verity.core.protocols.pouw.protocol import DomainError

    call = F.call(32, 64, 32)
    t, law = traced_as("ncp-v2-shift24", 32, 64, 32), PL.WorkLaw(2)
    first = _anchors(call)
    twin = Anchors(call.salt, call.forward, call.layer, call.site, call.weight + 1, tuple(call.rowids), tuple(call.colids))
    assert twin.call_index == first.call_index
    with pytest.raises(DomainError, match="repeated"):
        Receipt(((first, t.shape, hashlib.sha512(b"1").digest()), (twin, t.shape, hashlib.sha512(b"2").digest())))
    _receipt, window = _window(t, [first, _anchors(call, forward=call.forward + 1)])
    drawn, real = [], plan_module.draw
    monkeypatch.setattr(plan_module, "draw", lambda key, lay, law, index: drawn.append(index) or real(key, lay, law, index))
    t.draw(law, first, window)
    for again in (first, twin):
        with pytest.raises(DomainError, match="repeated|not on the window's receipt"):
            t.draw(law, again, window)
    assert drawn == [first.call_index] and window.seen == {first.call_index}
    t.draw(law, _anchors(call, forward=call.forward + 1), window)
    assert len(drawn) == 2
    for no_ledger in (None, set()):
        with pytest.raises(TypeError):
            t.draw(law, first, no_ledger)
    assert len(drawn) == 2


def test_the_window_key_is_derived_from_the_complete_receipt(monkeypatch):
    """Root's C1: the window's key comes after every call's commitment. A `Receipt` lists every call of the window with its
    shape and commitment, and is immutable; `AdmittedCalls.open` derives the key from it alone, under `DRAW_DOMAIN` with
    {"receipt": digest}, and `Traced.draw` refuses a ledger not opened from a receipt and a call the receipt doesn't list
    (another index, the same index with other anchors, or another shape), before `plan.draw` runs. The digest binds each
    call's full anchors (salt, weight id, rows and columns), shape and commitment, and not the order they are listed in."""
    import dataclasses
    import hashlib

    import pytest

    import circuit_fixtures as F
    from verity.core.protocols.pouw.circuit import anchors as A
    from verity.core.protocols.pouw.circuit import plan as plan_module
    from verity.core.protocols.pouw.circuit import traced_as
    from verity.core.protocols.pouw.protocol import DomainError

    call = F.call(32, 64, 32)
    t, law = traced_as("ncp-v2-shift24", 32, 64, 32), PL.WorkLaw(2)
    a, b, unlisted = (_anchors(call, forward=f) for f in (1, 2, 3))
    source = bytes(range(32))
    receipt, window = _window(t, [a, b], source)
    assert window.key == derive(source, PL.DRAW_DOMAIN, {"receipt": receipt.digest()})
    drawn, real = [], plan_module.draw
    monkeypatch.setattr(plan_module, "draw", lambda key, lay, law, index: drawn.append(index) or real(key, lay, law, index))
    other_shape = ncp2.Shape(32, 128, 32)
    twin = dataclasses.replace(a, weight=a.weight + 1)
    refusals = [(lambda: t.draw(law, a, A.AdmittedCalls()), "opened from its complete receipt"),
                (lambda: t.draw(law, unlisted, window), "not on the window's receipt"),
                (lambda: t.draw(law, twin, window), "not on the window's receipt"),
                (lambda: window.admit(a, other_shape), "not on the window's receipt")]
    for refused, match in refusals:
        with pytest.raises(DomainError, match=match):
            refused()
    assert drawn == []
    t.draw(law, a, window)
    t.draw(law, b, window)
    assert drawn == [a.call_index, b.call_index]
    ca, cb = (c for _an, _s, c in receipt.calls)
    assert A.Receipt(((b, t.shape, cb), (a, t.shape, ca))).digest() == receipt.digest()
    changed = [((a, t.shape, ca), (b, t.shape, hashlib.sha512(b"b'").digest())), ((a, t.shape, ca), (b, other_shape, cb))]
    for field, value in (("salt", bytes(32)), ("weight", b.weight + 1), ("rows", tuple(reversed(b.rows))),
                         ("cols", tuple(reversed(b.cols)))):
        changed.append(((a, t.shape, ca), (dataclasses.replace(b, **{field: value}), t.shape, cb)))
    assert len({A.Receipt(c).digest() for c in changed} | {receipt.digest()}) == len(changed) + 1
    for bad in ((), ((a, t.shape, b"short"),)):
        with pytest.raises(DomainError):
            A.Receipt(bad)
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.calls = ()
    with pytest.raises(TypeError):
        A.AdmittedCalls.open([(a, t.shape, ca)])


def test_a_caller_built_ledger_has_no_key():
    """The red team's first probe: a ledger built with a key from a caller-chosen source. `AdmittedCalls` takes no receipt, key
    or source, and sets none afterwards, so only `AdmittedCalls.open` makes one that draws."""
    import pytest

    import circuit_fixtures as F
    from verity.core.protocols.pouw.circuit import traced_as
    from verity.core.protocols.pouw.circuit.anchors import AdmittedCalls
    from verity.core.protocols.pouw.protocol import DomainError

    call = F.call(32, 64, 32)
    t, an = traced_as("ncp-v2-shift24", 32, 64, 32), _anchors(call)
    receipt, _window_ = _window(t, [an])
    chosen = derive(b"caller-chosen", PL.DRAW_DOMAIN, {"receipt": receipt.digest()})
    for built in (lambda: AdmittedCalls(receipt=receipt, key=chosen), lambda: AdmittedCalls(receipt, chosen)):
        with pytest.raises(TypeError):
            built()
    bare = AdmittedCalls()
    for name, value in (("key", chosen), ("_key", chosen), ("receipt", receipt), ("_receipt", receipt), ("_source", b"x")):
        with pytest.raises(AttributeError):
            setattr(bare, name, value)
    with pytest.raises(DomainError, match="opened from its complete receipt"):
        t.draw(PL.WorkLaw(2), an, bare)


def test_an_opened_ledger_s_key_cannot_be_reassigned():
    """The red team's second probe: an opened ledger whose key is reassigned. The key is set once, in `AdmittedCalls.open`, and
    `Traced.draw` reads it from the ledger, so a caller's key never reaches a draw."""
    import pytest

    import circuit_fixtures as F
    from verity.core.protocols.pouw.circuit import traced_as

    call = F.call(32, 64, 32)
    t, an = traced_as("ncp-v2-shift24", 32, 64, 32), _anchors(call)
    receipt, window = _window(t, [an])
    derived, chosen = window.key, derive(b"caller-chosen", PL.DRAW_DOMAIN, {"receipt": receipt.digest()})
    for name in ("key", "_key"):
        with pytest.raises(AttributeError):
            setattr(window, name, chosen)
    assert window.key == derived and t.draw(PL.WorkLaw(2), an, window) == PL.draw(derived, t.layout(), PL.WorkLaw(2),
                                                                                  an.call_index)


def test_an_opened_ledger_s_receipt_cannot_be_swapped():
    """The red team's third probe: an opened ledger whose receipt is swapped for a larger one after the key exists, so a
    call committed after the key would be drawn under it. The receipt is set once, in `AdmittedCalls.open`, and the later call
    is refused."""
    import hashlib

    import pytest

    import circuit_fixtures as F
    from verity.core.protocols.pouw.circuit import traced_as
    from verity.core.protocols.pouw.circuit.anchors import Receipt
    from verity.core.protocols.pouw.protocol import DomainError

    call = F.call(32, 64, 32)
    t, a, late = traced_as("ncp-v2-shift24", 32, 64, 32), _anchors(call, forward=1), _anchors(call, forward=2)
    receipt, window = _window(t, [a])
    larger = Receipt(receipt.calls + ((late, t.shape, hashlib.sha512(b"late").digest()),))
    for name in ("receipt", "_receipt"):
        with pytest.raises(AttributeError):
            setattr(window, name, larger)
    assert window.receipt is receipt
    with pytest.raises(DomainError, match="not on the window's receipt"):
        t.draw(PL.WorkLaw(2), late, window)


def test_openings_are_checked_against_the_receipt_s_commitment():
    """The red team's tier-3 item: the call's A-row openings are checked against its commitment as the opened ledger's
    receipt lists it, by call index (`AdmittedCalls.commitment`), never one supplied with the openings. The prover's own rows
    under the receipt's commitment pass; rows committed under any other commitment, a missing row, a row opened twice,
    or a call the receipt doesn't list are refused."""
    import pytest

    import circuit_fixtures as F
    from verity.core.protocols.pouw import audit
    from verity.core.protocols.pouw.circuit import traced_as
    from verity.core.protocols.pouw.circuit.anchors import AdmittedCalls
    from verity.core.protocols.pouw.protocol import DomainError

    call = F.call(32, 64, 32)
    t, an = traced_as("ncp-v2-shift24", 32, 64, 32), _anchors(call)
    rows = [bytes(v & 0xFF for v in r) for r in call.a]
    binding = audit.activation_binding(call.salt, 7)
    domain = audit.rows_domain(binding, 7, len(rows))
    tree = audit.commit_rows(binding, 7, rows)
    other = audit.commit_rows(binding, 7, [bytes(len(r)) for r in rows])
    _receipt, window = _window(t, [an], commitments=[tree.commitment.root])
    assert window.commitment(an) == tree.commitment.root
    plan = t.draw(PL.WorkLaw(2), an, window)
    opened = sorted(r for kind, r in plan.openings if kind == "a")
    t.check_openings(plan, an, window, domain, [tree.open(r) for r in opened])
    for bad in ([other.open(r) for r in opened], [tree.open(r) for r in opened[1:]],
                [tree.open(r) for r in opened] + [tree.open(opened[0])]):
        with pytest.raises(DomainError):
            t.check_openings(plan, an, window, domain, bad)
    for no_receipt in (AdmittedCalls(),):
        with pytest.raises(DomainError, match="window's receipt"):
            no_receipt.commitment(an)
    with pytest.raises(DomainError, match="not on the window's receipt"):
        window.commitment(_anchors(call, forward=call.forward + 9))


def test_the_record_keeps_the_source_once_the_draw_is_sent():
    """The red team's answer to question 2: the audit record keeps the window's source and receipt digest once its draw
    is sent, so a third party re-derives the key (`replay_key`) and so the draw. Before it is sent the record is refused,
    and after it the window draws nothing more. What the record can't show is that the source was fresh."""
    import pytest

    import circuit_fixtures as F
    from verity.core.protocols.pouw.circuit import traced_as
    from verity.core.protocols.pouw.circuit.anchors import replay_key
    from verity.core.protocols.pouw.protocol import DomainError

    call = F.call(32, 64, 32)
    t, a, b = traced_as("ncp-v2-shift24", 32, 64, 32), _anchors(call, forward=1), _anchors(call, forward=2)
    source = bytes(range(32))
    receipt, window = _window(t, [a, b], source)
    plan = t.draw(PL.WorkLaw(2), a, window)
    with pytest.raises(DomainError, match="once its draw is sent"):
        window.record()
    window.sent()
    with pytest.raises(DomainError, match="sent"):
        t.draw(PL.WorkLaw(2), b, window)
    record = window.record()
    assert record == {"domain": PL.DRAW_DOMAIN, "receipt": receipt.digest().hex(), "source": source.hex()}
    assert replay_key(record) == window.key
    assert PL.draw(replay_key(record), t.layout(), PL.WorkLaw(2), a.call_index) == plan
    with pytest.raises(DomainError):
        replay_key({**record, "domain": "verity/pouw/other/v1"})


def test_each_window_s_source_is_fresh():
    """Root's C2: A4 models one fresh uniform source per derivation. `AdmittedCalls.open` takes no source: it draws 32 fresh bytes
    from the operating system's CSPRNG (`secrets`) for each window, once the receipt is complete, so two windows, even
    over one receipt, get independent keys, and no window's key is a function of another's."""
    import inspect

    import circuit_fixtures as F
    from verity.core.protocols.pouw.circuit import anchors as A
    from verity.core.protocols.pouw.circuit import traced_as

    call = F.call(32, 64, 32)
    t = traced_as("ncp-v2-shift24", 32, 64, 32)
    receipt, _window_ = _window(t, [_anchors(call)])
    assert list(inspect.signature(A.AdmittedCalls.open).parameters) == ["receipt"]
    assert len(A._fresh_source()) == A.SOURCE_BYTES == 32
    keys = {A.AdmittedCalls.open(receipt).key for _ in range(8)}
    assert len(keys) == 8


def test_the_call_s_y_is_linked_to_the_weight_program():
    """The call reads its weight's Y strips and c₀ as inputs, so the plan opens them at each drawn tile's column strip
    beside the weight Program's outputs there, and the verifier compares the two (`Traced.check_link`, X-SPC-95). Another
    weight's Y, or one c₀ changed, is refused."""
    import pytest

    import circuit_fixtures as F
    from verity.ml.circuits.evaluation import evaluate
    from verity.core.protocols.pouw.circuit import ncp2, traced_as
    from verity.core.protocols.pouw.protocol import DomainError

    call = F.call(32, 64, 32)
    t = traced_as("ncp-v2-shift24", 32, 64, 32)
    _receipt, window = _window(t, [_anchors(call)])
    plan = t.draw(PL.WorkLaw(2), _anchors(call), window)
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
