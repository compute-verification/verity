"""Checker v2 export (`gkr_export.v2`): census against checker_min.md, honest rows accepted by the circuit's own
evaluator, every hint / lookup-output mutation rejected, the chain spec and the implicit tables."""
from __future__ import annotations

import random
from collections import Counter

import pytest

from verity_numerical.checker import REAL
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.gadgets import Epilogue
from verity_numerical.checker.v2.gadgets import Pack
from verity_numerical.checker.v2.step import gen_unit, vu_output, zero_state
from verity_numerical.checker.v2.tables import shift_table, tnorm_table
from verity_numerical.gkr_export import v2

STEP = 16


@pytest.fixture(scope="module")
def circuits():
    return v2.build_unit_circuit_v2(), v2.build_epilogue_circuit_v2()


def _ops(rng, n):
    bf = lambda: (rng.randrange(2) << 15) | (rng.randrange(100, 150) << 7) | rng.randrange(128)  # noqa: E731
    return [bf() for _ in range(n)], [bf() for _ in range(n)]


def test_census_matches_checker_min(circuits):
    (u, up), (e, ep) = circuits
    prods = sum(1 for w in u.wires if w[0] == "prod")
    by_table = Counter(t for t, _ in u.queries)
    # checker_min.md 3.1, VU body (state_decode's 3 mul / 4 lin / 5 lookups and Pack's 4 mul are not in a unit):
    # 33 - 3 - 4 = 26 products; 89 - 5 = 84 lookups plus the two one-sided halves of each group's `attained`
    assert prods == 26
    assert len(u.queries) == 86
    assert by_table == {"T_OP": 32, "ALIGN4": 18, "SHIFT": 16, "SSHIFT_HI": 2, "SSHIFT_LO": 2, "LEAD": 4, "LEADNORM": 2,
                        "TNORM": 2, "R12": 4, "R4": 4}
    assert max(u.depth) == 1                     # every product feeds a materialised key or an output assertion
    assert len(up) == 12                          # 2 groups x (E, M_hi, M_lo, sgn, v_hi, v_lo)
    assert len(u.cols) == 264
    assert u.cols[1:5] == ["x.s", "x.e", "x.M", "x.z"] and u.cols[37:41] == ["y.s", "y.e", "y.M", "y.z"]
    assert len(e.cols) == 16 and len(ep) == 6
    links = v2.chain_spec(u, e)
    assert [lk[3] for lk in links[:4]] == list(zero_state(REAL))
    assert sum((97 if lk[0] else 1 + (95 if lk[4] else 0)) for lk in links) + 1 == 583   # per VU at K = 1536


def test_honest_rows_accepted_and_every_mutation_rejected(circuits):
    unit, epi = circuits
    uctx, upaths = unit
    rng = random.Random(3)
    a, b = _ops(rng, 2 * STEP)
    y16 = vu_output(REAL, a, b)
    rows, erow = v2.vu_rows_v2(unit, epi, a, b, y16)
    assert len(rows) == 2 and v2.verify_row(uctx, rows[1]) is None and v2.verify_row(epi[0], erow) is None
    # mutate every hint and every lookup-output column of unit 1 by +1: the circuit must reject each
    base = 9 + 2 * STEP
    hint_cols = list(range(base, base + len(upaths)))
    out_cols = [i for i, d in enumerate(uctx.derived) if d is not None and d[0] == "row"]
    for col in hint_cols + out_cols:
        bad = list(rows[1])
        bad[col] += 1
        assert v2.verify_row(uctx, bad) is not None, uctx.cols[col]
    # a wrong outgoing state component
    for col in range(37, 41):
        bad = list(rows[1])
        bad[col] += 1
        assert v2.verify_row(uctx, bad) is not None, uctx.cols[col]


def test_saturated_unit_is_a_domain_reject(circuits):
    unit, epi = circuits
    a = [0x7F7F] * (2 * STEP)
    b = [0x7F7F] * (2 * STEP)
    h, s_out, flags = gen_unit(REAL, zero_state(REAL), a[:STEP], b[:STEP])
    assert any(o for o, _ in flags)
    with pytest.raises(Exception, match="saturation before the last unit"):
        v2.vu_rows_v2(unit, epi, a, b, 0x7F80)
    # a single saturating unit is fine at the end: the epilogue packs the infinity
    y32 = Pack.check(IntCtx(), REAL, s_out, flags)
    assert y32 == 0x7F800000
    y16 = vu_output(REAL, a[:STEP], b[:STEP])
    assert y16 == Epilogue.check(IntCtx(), REAL, y32, Epilogue.gen(REAL, y32))
    rows, erow = v2.vu_rows_v2(unit, epi, a[:STEP], b[:STEP], y16)
    assert len(rows) == 1 and erow[5] == 1


def test_implicit_table_headers_match_rows():
    for t in (shift_table(REAL), tnorm_table(REAL)):
        assert v2.implicit_table_header(t) is not None
        rng = random.Random(t.rows)
        for key in [0, t.hi - 1] + [rng.randrange(t.hi) for _ in range(200)]:
            assert v2._implicit_row(t.name, key, REAL) == t.row(key)
