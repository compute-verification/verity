"""The candidate generator (`candidates`): its dynamic program against every pure cut, the size classes' pruning, re-bound
statics, the hints, and the shape it reads off a Definition's body."""

from __future__ import annotations

import pytest

from verity.core.protocols.sampled_proofs.candidates import Shape, _lams, _Program, cuts, pressure, shape_of
from verity.core.protocols.sampled_proofs.one_stage.partition import UNIT_CLASS_MAX
from verity.core.protocols.sampled_proofs.optimize import CostModel, Objective, frontier

STEP = Shape("Step", 4190, 32, 32, rows=(32, 32, 4, 4), commit_bits=32, cap=4096)
CHUNK = Shape("Dot{K=256}", 8 * 4190, 256, 32, rows=(256, 256, 4, 4), commit_bits=32, cap=4096, parts=((STEP, 8),))
DOT = Shape("Dot{K=1024}", 32 * 4190, 1024, 16, rows=(1024, 1024, 2), commit_bits=16, cap=4096, parts=((STEP, 32),),
            ways=(("K=256", ((CHUNK, 4),)),))
N = 10 ** 6
OBJECTIVE = Objective(2.0 ** -40)


def test_program_takes_the_cheapest_pure_cut_at_every_density():
    costs = CostModel(per_and=1e-10, statement_fixed=1.0, staged_per_and=1e-10, commit_per_bit=3e-9)
    pure = {"Dot{K=1024}": [(DOT, N)], "Dot{K=256}": [(CHUNK, 4 * N)], "Step": [(STEP, 32 * N)]}
    for lam in _lams(DOT, N, OBJECTIVE, 21):
        prog = _Program("dot", costs, OBJECTIVE, lam, None)
        price, units = prog.best(DOT, N)
        prices = {k: sum(prog.unit(s, m) for s, m in v) for k, v in pure.items()}
        assert price == pytest.approx(min(prices.values()), rel=1e-12)
        assert prices[units[0][0].id] == pytest.approx(price, rel=1e-12)


def test_cuts_are_alternatives_the_planner_chooses_among():
    costs = CostModel(per_and=1e-10, commit_per_bit=1.1e-10)
    cands, alts, _ = cuts("dot", DOT, N, costs, OBJECTIVE, grid=41)
    assert set(alts.values()) == {(("Dot{K=1024}", N),), (("Dot{K=256}", 4 * N),), (("Step", 32 * N),)}
    assert {c.cut for c in cands} == set(alts) and len({c.template for c in cands}) == len(cands)
    plans = frontier(cands, costs, OBJECTIVE, grid=41, exact=False)
    assert {alts[p.choice["dot"]][0][0] for p in plans} >= {"Dot{K=1024}", "Dot{K=256}"}


def test_size_classes_admit_rows_and_tiles_and_prune_past_the_widest():
    coord = Shape("Coord", 32 * 4190, 1024, 32, rows=(1024, 1024, 4), commit_bits=32, cap=4096)
    tile = Shape("Tile2", 2 * 32 * 4190, 2048, 64, out_gates=2, rows=(1024, 1024, 1024, 8), commit_bits=64, cap=4096,
                 parts=((coord, 2),))
    costs = CostModel(per_and=1e-10, statement_fixed=1.0)
    _, alts, hints = cuts("tile", tile, N, costs, OBJECTIVE, grid=21)
    assert (("Tile2", N),) in alts.values() and not [h for h in hints if h.kind == "width"]
    row = Shape("Row{N=128}", 128 * 4190, 128 * 1024, 128 * 16, out_gates=128, rows=(2048, 256), commit_bits=2048,
                cap=4096, parts=((coord, 128),))
    _, alts, _ = cuts("row", row, N, costs, OBJECTIVE, grid=21)
    assert (("Row{N=128}", N),) in alts.values()
    n = UNIT_CLASS_MAX // 32 + 1
    wide = Shape(f"Row{{N={n}}}", n * 4190, n * 1024, 32 * n, out_gates=n, rows=(4 * n,), commit_bits=32 * n,
                 cap=4096, parts=((coord, n),))
    _, alts, _ = cuts("wide", wide, N, costs, OBJECTIVE, grid=21)
    assert alts == {"c0": (("Coord", n * N),)}
    with pytest.raises(ValueError, match="fit no size class"):
        cuts("wide", Shape("Wide", 1, 1, UNIT_CLASS_MAX + 1, out_gates=2), N, CostModel(per_and=1.0), OBJECTIVE)


def test_boundary_pressure_names_the_unit_between_two_definitions():
    """With commitment against proving balanced at sqrt(c0 / (lam alpha)) between 32 and 1024 MACs, and no chunk
    Definition to bind, proof blocks of an eighth of the dot beat both the dot and its steps."""
    dot = Shape("Dot{K=1024}", 32 * 4190, 1024, 16, rows=(1024, 1024, 2), commit_bits=16, cap=4096, parts=((STEP, 32),))
    _, alts, hints = cuts("dot", dot, N, CostModel(per_and=1e-10, commit_per_bit=1.1e-10), OBJECTIVE, grid=41)
    assert set(alts.values()) == {(("Dot{K=1024}", N),), (("Step", 32 * N),)}
    (h,) = hints
    assert (h.kind, h.node, h.child, h.split) == ("boundary", "Dot{K=1024}", "Step", 8) and h.saving > 0.2
    assert pressure("dot", dot, N, CostModel(per_and=1e-10, commit_per_bit=1.1e-10), OBJECTIVE, h.lam) == [h]
    assert pressure("dot", dot, N, CostModel(per_and=1e-10, commit_per_bit=1.1e-10), OBJECTIVE, h.lam * 1e-6) == []


def test_shape_of_reads_a_definitions_body():
    """bf16 GemmCoordinate at K = 64: four pinned Hopper steps and the rounding, its rows its operands and its return."""
    import json
    from pathlib import Path

    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import gemm, prims

    pins = json.loads((Path(prims.__file__).parent / "prims.circuit-check.json").read_text())["pieces"]
    pieces = {k: v["and"] for k, v in pins.items()}
    coord = bind(gemm.GemmCoordinate, K=64, DOT=prims.HopperBF16WgmmaDot16)
    s = shape_of(coord, pieces, work={"HopperBF16WgmmaDot16_v1": 16})
    step, rounding = pieces["HopperBF16WgmmaDot16_v1"], pieces["F2fpBf16_v1"]
    assert (s.ands, s.work, s.out_bits, s.rows) == (4 * step + rounding, 64, 16, (128, 128, 2))
    (dot, one), _ = s.parts
    assert one == 1 and [(c.id, k) for c, k in dot.parts] == [("Const32[0x00000000]_v1", 1),
                                                             ("HopperBF16WgmmaDot16_v1", 4)]
    with pytest.raises(KeyError, match="no AND count"):
        shape_of(coord, {})
