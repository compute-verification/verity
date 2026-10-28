"""orbit.py against the plain int8 GEMM, bit for bit, on operands whose forms overflow int8 often (every carry path) and
never (the uncorrected orbit is then exact too)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import orbit  # noqa: E402


def _ops(seed: int, m: int, n: int, k: int, lo: int = -128, hi: int = 127) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (rng.integers(lo, hi + 1, (m, k), dtype=np.int8), rng.integers(lo, hi + 1, (n, k), dtype=np.int8))


@pytest.mark.parametrize("m,n,k", [(2, 2, 2), (8, 6, 10), (64, 96, 128), (130, 34, 66)])
@pytest.mark.parametrize("seed", [0, 1])
def test_exact_on_uniform_int8(m, n, k, seed):
    a, b = _ops(seed, m, n, k)
    pb = orbit.prepare_b(b)
    assert np.array_equal(orbit.orbit_mm(a, pb), orbit.plain_mm(a, b))


def test_carries_are_exercised():
    a, b = _ops(3, 64, 64, 64)
    _, a_carry, _ = orbit.forms_a(a)
    pb = orbit.prepare_b(b)
    for c in (a_carry, pb.carry):
        assert set(np.unique(c)) == {-1, 0, 1}
    assert not np.array_equal(orbit.orbit_mm(a, pb, corrected=False), orbit.plain_mm(a, b))


def test_extremes():
    """Every entry at an int8 extreme: the forms reach −256 and 254, and O's cells reach ±128·128·K."""
    k = 64
    for va, vb in [(-128, -128), (-128, 127), (127, 127)]:
        a = np.full((4, k), va, np.int8)
        b = np.full((6, k), vb, np.int8)
        assert np.array_equal(orbit.orbit_mm(a, orbit.prepare_b(b)), orbit.plain_mm(a, b))


def test_no_carry_below_64():
    """|x|, |w| ≤ 63: no form leaves int8, so the products on the wrapped forms alone are exact (the GPU's carry-free gate)."""
    a, b = _ops(4, 96, 64, 256, -63, 63)
    pb = orbit.prepare_b(b)
    assert not pb.carry.any()
    assert np.array_equal(orbit.orbit_mm(a, pb, corrected=False), orbit.plain_mm(a, b))


def test_int32_wraparound():
    """Large K near the s32 edge: sums past 2^31 in the assembly's partial sums wrap and still give O."""
    k = 8192
    a = np.full((2, k), -128, np.int8)
    b = np.full((2, k), -128, np.int8)
    b[1] = 127
    assert np.array_equal(orbit.orbit_mm(a, orbit.prepare_b(b)), orbit.plain_mm(a, b))


def test_layout():
    """B' holds the wrapped forms side by side along K in ORDER; carry_list names every nonzero carry."""
    _, b = _ops(5, 2, 8, 12)
    pb = orbit.prepare_b(b)
    h = pb.k // 2
    for s, p in enumerate(orbit.ORDER):
        f = orbit.form(b, orbit.B_FORMS[p])
        w, c = orbit.wrap(f)
        assert np.array_equal(pb.wide[:, s * h:(s + 1) * h], w)
        assert np.array_equal(pb.full[s], f)
        lst = pb.carry_list(s)
        assert len(lst) == np.count_nonzero(c)
        assert all(c[n, kk] == v for n, kk, v in lst)


def test_three_accumulators_is_the_minimum():
    assert orbit.min_live_accumulators() == 3


def test_odd_shapes_refused():
    with pytest.raises(ValueError, match="even"):
        orbit.prepare_b(np.zeros((3, 4), np.int8))
