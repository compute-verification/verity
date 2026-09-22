"""The VU export: the epilogue circuit is ``Epilogue.check`` verbatim, chained unit rows satisfy the unit circuit and link
through their words, and a wrong output word or a broken chain is caught."""
from __future__ import annotations

import random

import pytest
from verity_numerical.checker import REAL, Reject
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.gadgets import Epilogue
from verity_numerical.checker.step import step_output
from verity_numerical.gkr_export import build_unit_circuit
from verity_numerical.gkr_export.circuit import verify_unit
from verity_numerical.gkr_export.vu import (
    STEP,
    build_epilogue_circuit,
    epilogue_witness,
    vu_rows,
)


def _bf16(rng: random.Random) -> int:
    return (rng.randrange(2) << 15) | (rng.randrange(100, 150) << 7) | rng.randrange(2**7)


def _chain(rng: random.Random, steps: int) -> tuple[list[int], list[int], list[int], int]:
    a = [_bf16(rng) for _ in range(steps * STEP)]
    b = [_bf16(rng) for _ in range(steps * STEP)]
    accs, c = [], 0
    for i in range(steps):
        c = step_output(REAL, c, a[i * STEP:(i + 1) * STEP], b[i * STEP:(i + 1) * STEP])
        accs.append(c)
    y16 = Epilogue.check(IntCtx(), REAL, c, Epilogue.gen(REAL, c))
    return a, b, accs, y16


def test_epilogue_circuit_shape() -> None:
    ctx, paths = build_epilogue_circuit()
    assert ctx.cols[:3] == ["one", "y32", "y16"]
    assert paths == ["h.hi", "h.lo", "h.hi2", "h.lsb", "h.carry", "h.rem"]
    assert sum(1 for w in ctx.wires if w[0] == "prod") == 2           # the two booleans
    assert len(ctx.asserts) == 6 and len(ctx.queries) == 5            # 3 eq + 2 boolean; 5 range chunks into R16
    assert all(t == "R16" for t, _ in ctx.queries)


def test_chained_rows_satisfy_both_circuits_and_link() -> None:
    unit, epi = build_unit_circuit(), build_epilogue_circuit()
    rng = random.Random(3)
    for steps in (1, 2, 4):
        a, b, accs, y16 = _chain(rng, steps)
        rows, erow = vu_rows(unit, epi, a, b, y16, accs)
        assert len(rows) == steps
        c = 0
        for r in rows:
            assert verify_unit(unit[0], r) is None
            assert r[1] == c            # x.c is the previous word (0 first)
            c = r[2 + 2 * STEP]         # y
        assert verify_unit(epi[0], erow) is None
        assert erow[1] == c and erow[2] == y16
        # without recorded accumulators the checker's own chain is used and agrees
        rows2, erow2 = vu_rows(unit, epi, a, b, y16)
        assert rows2 == rows and erow2 == erow


def test_wrong_word_and_wrong_record_are_rejected() -> None:
    unit, epi = build_unit_circuit(), build_epilogue_circuit()
    rng = random.Random(5)
    a, b, accs, y16 = _chain(rng, 2)
    with pytest.raises(Reject):
        vu_rows(unit, epi, a, b, y16 ^ 1, accs)
    bad = list(accs)
    bad[0] ^= 1 << 3
    with pytest.raises(Reject):
        vu_rows(unit, epi, a, b, y16, bad)
    # the epilogue circuit itself rejects a wrong output word with honest hints
    ectx, epaths = epi
    row = epilogue_witness(ectx, epaths, accs[-1], y16 ^ 1, Epilogue.gen(REAL, accs[-1]))
    assert verify_unit(ectx, row) == "assert y16"
