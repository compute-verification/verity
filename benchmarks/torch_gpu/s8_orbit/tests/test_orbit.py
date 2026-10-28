"""orbit.py against the plain int8 GEMM: exact at every carry pattern, its carry lists, and what holds without corrections."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import orbit as O  # noqa: E402


def ints(rng, shape, lo=-128, hi=127):
    return rng.integers(lo, hi + 1, shape).astype(np.int8)


@pytest.mark.parametrize("m,n,k", [(64, 96, 128), (33, 50, 71), (2, 2, 2), (1, 3, 5)])
def test_orbit_is_the_plain_gemm(m, n, k):
    rng = np.random.default_rng(m * 1000 + n + k)
    x, w = ints(rng, (m, k)), ints(rng, (n, k))
    assert np.array_equal(O.orbit(x, w, tile_n=16), O.plain(x, w))


def test_every_form_carries_at_the_extremes():
    rng = np.random.default_rng(7)
    x = np.where(rng.random((48, 64)) < 0.5, -128, 127).astype(np.int8)
    w = np.where(rng.random((40, 64)) < 0.5, -128, 127).astype(np.int8)
    _, ca = O.a_forms(x)
    _, cb = O.b_forms(w)
    assert all(ca[j].any() for j in (0, 1, 4, 5, 6)) and all(cb[j].any() for j in (0, 2, 3, 5, 6))
    assert np.array_equal(O.orbit(x, w, tile_n=8), O.plain(x, w))


def test_extreme_sums_wrap_with_unit_carries():
    f = np.array([-256, -255, -129, -128, 127, 128, 254, 255], np.int16)
    w, c = O.wrap(f)
    assert np.array_equal(w.astype(np.int16) + 256 * c.astype(np.int16), f)
    assert np.array_equal(c, [-1, -1, -1, 0, 0, 1, 1, 1])


def test_carry_lists_round_trip():
    rng = np.random.default_rng(3)
    _, cb = O.b_forms(ints(rng, (200, 90)))
    cl = O.b_carries(cb, tile_n=96)
    assert cl.offsets.shape == (7, 3) and cl.offsets[-1, -1] == len(cl.entries) == np.count_nonzero(cb)
    for j in range(7):
        assert np.array_equal(O.carry_matrix(cl, j), cb[j])
        for t in range(2):
            k = cl.entries[cl.offsets[j, t]:cl.offsets[j, t + 1]] >> O.K_SHIFT
            assert np.all(np.diff(k.astype(np.int64)) >= 0)


def test_no_carries_below_64():
    rng = np.random.default_rng(5)
    x, w = ints(rng, (64, 256), -64, 63), ints(rng, (96, 256), -64, 63)
    assert not O.a_forms(x)[1].any() and not O.b_forms(w)[1].any()
    assert np.array_equal(O.wrapped_only(x, w), O.plain(x, w))


def test_uncorrected_orbit_is_exact_mod_256():
    rng = np.random.default_rng(11)
    x, w = ints(rng, (32, 64)), ints(rng, (48, 64))
    got, want = O.wrapped_only(x, w), O.plain(x, w)
    assert not np.array_equal(got, want)
    assert np.array_equal(got.astype(np.int64) & 0xFF, want.astype(np.int64) & 0xFF)
