"""The circuit export agrees with WP2: honest random units satisfy every gate/lookup, every exportable negative fails one."""
from __future__ import annotations

import random

from verity_numerical.gkr_export import (
    build_unit_circuit,
    random_finite_unit,
    unit_witness,
)
from verity_numerical.gkr_export.circuit import verify_unit
from verity_numerical.gkr_export.negatives import negative_cases


def test_census_matches_the_checker() -> None:
    ctx, _ = build_unit_circuit()
    assert sum(1 for w in ctx.wires if w[0] == "prod") == 358          # the WP2 census's mul count
    assert max(ctx.depth) == 4


def test_honest_units_satisfy_the_circuit() -> None:
    ctx, paths = build_unit_circuit()
    rng = random.Random(7)
    for _ in range(50):
        c, a, b, y, h = random_finite_unit(rng)
        assert verify_unit(ctx, unit_witness(ctx, paths, c, a, b, y, h)) is None


def test_every_negative_fails_the_circuit() -> None:
    ctx, paths = build_unit_circuit()
    cases = negative_cases()
    assert len(cases) >= 50
    for name, c, a, b, y, h, _pred in cases:
        assert verify_unit(ctx, unit_witness(ctx, paths, c, a, b, y, h)) is not None, name
