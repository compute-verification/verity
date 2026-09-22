"""The BabyBear export agrees with the embedded checker: honest units pass, every WP2 negative and every step-level
battery case fails, and the census is what ``PROTOCOL.md`` section 13 states."""
from __future__ import annotations

import random
from collections import Counter

import pytest
from verity_numerical.checker import REAL
from verity_numerical.gkr_export.babybear import (
    BABYBEAR,
    build_epilogue_circuit_bb,
    build_unit_circuit_bb,
    check_embedded,
    epilogue_witness_bb,
    random_finite_unit_bb,
    unit_witness_bb,
    verify_unit_bb,
)
from verity_numerical.gkr_export.negatives import negative_cases


@pytest.fixture(scope="module")
def unit():
    return build_unit_circuit_bb()


def test_census(unit) -> None:
    ctx, _ = unit
    assert len(ctx.cols) == 444
    assert sum(1 for w in ctx.wires if w[0] == "prod") == 322          # 358 - 36 lt_pow products
    assert len(ctx.asserts) == 322
    # 360 range chunks (exact-width tables R<w>: the shifted single-chunk form into R16 was forgeable), 22 table rows
    assert Counter(t for t, _ in ctx.queries) == {
        "R16": 139, "R14": 10, "R9": 36, "R8": 137, "R7": 38, "ALIGN": 16, "POW": 2, "LEAD": 2, "NORM": 2,
    }
    assert max(ctx.depth) == 4
    assert ctx.tables["NORM"].outs == ("mul", "div", "idiv", "mbnd")       # vbnd dropped
    assert ctx.tables["LEAD"].outs == ("lo", "ispan", "span")
    assert ctx.cols[:3] == ["one", "x.c_lo", "x.c_hi"] and ctx.cols[35:37] == ["y_lo", "y_hi"]


def test_honest_units_satisfy_the_circuit(unit) -> None:
    ctx, paths = unit
    rng = random.Random(11)
    for _ in range(30):
        c, a, b, y, h = random_finite_unit_bb(rng)
        check_embedded(REAL, c, a, b, y, h)
        vals = unit_witness_bb(ctx, paths, c, a, b, y, h)
        assert all(0 <= v < BABYBEAR for v in vals)
        assert verify_unit_bb(ctx, vals) is None


def test_every_negative_fails_the_circuit(unit) -> None:
    ctx, paths = unit
    cases = negative_cases()
    assert len(cases) >= 50
    for name, c, a, b, y, h, _pred in cases:
        assert verify_unit_bb(ctx, unit_witness_bb(ctx, paths, c, a, b, y, h)) is not None, name


def test_step_battery_cases_fail_the_circuit(unit) -> None:
    from verity_numerical.redteam.battery_babybear import load_battery

    ctx, paths = unit
    n = 0
    for case in load_battery():
        if case.level != "step":
            continue
        x = case.inputs
        vals = unit_witness_bb(ctx, paths, x["c"], x["a"], x["b"], case.claimed["y"], case.hints)
        assert verify_unit_bb(ctx, vals) is not None, case.id
        n += 1
    assert n >= 60


def test_epilogue_circuit() -> None:
    from verity_numerical.checker.ctx import IntCtx
    from verity_numerical.checker.gadgets import Epilogue

    ectx, epaths = build_epilogue_circuit_bb()
    assert ectx.cols == ["one", "y32_lo", "y32_hi", "y16", "h.hi2", "h.lsb", "h.carry", "h.rem"]
    for y32 in (0x4077199C, 0x3F808000, 0x3F818000, 0xFF7FFFFF, 0):
        he = Epilogue.gen(REAL, y32)
        y16 = Epilogue.check(IntCtx(), REAL, y32, he)
        assert verify_unit_bb(ectx, epilogue_witness_bb(ectx, epaths, y32, y16, he)) is None
        assert verify_unit_bb(ectx, epilogue_witness_bb(ectx, epaths, y32, y16 ^ 1, he)) is not None
