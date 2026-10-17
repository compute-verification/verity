import itertools

import numpy as np
import pytest
from exact_kit import cube


def _coeff_brute(vals, n):
    """c_m = sum over subsets s of m of (-1)^(|m| - |s|) f(s)."""
    out = []
    for m in range(1 << n):
        subs = [s for s in range(1 << n) if s & ~m == 0]
        out.append(sum((-1) ** (bin(m).count("1") - bin(s).count("1")) * int(vals[s]) for s in subs))
    return np.array(out, dtype=object)


@pytest.mark.parametrize("n", range(0, 7))
def test_mobius_against_the_subset_sum(n):
    rng = np.random.default_rng(n)
    vals = rng.integers(-1000, 1000, 1 << n)
    want = _coeff_brute(vals, n)
    assert cube.mobius(vals, n).tolist() == list(want)
    for bits in (1, 8, 31, 32, 33, 64):
        got = cube.mobius(vals, n, bits=bits)
        assert got.dtype == (np.uint32 if bits <= 32 else np.uint64)
        assert [int(x) for x in got] == [int(w) % (1 << bits) for w in want]
    assert cube.evaluate(cube.mobius(vals, n), n).tolist() == vals.tolist()
    assert [int(x) for x in cube.evaluate(cube.mobius(vals, n, 32), n, 32)] == [int(v) % (1 << 32) for v in vals]


def test_mobius_reads_a_polynomial_back():
    n = 6
    idx = cube.points(n)
    x = [cube.bit(idx, i).astype(np.int64) for i in range(n)]
    f = 3 * x[0] * x[2] - 5 * x[1] * x[3] * x[5] + 7 * x[4] + 11
    c = cube.mobius(f, n)
    want = {0: 11, 0b10000: 7, 0b101: 3, 0b101010: -5}
    assert {m: int(v) for m, v in enumerate(c) if v} == want
    assert cube.degree(n)[0b101010] == 3
    w32 = (f.astype(np.int64) % (1 << 32)).astype(np.uint32)
    assert int(cube.mobius(w32, n, 32)[0b101010]) == (1 << 32) - 5
    rows = cube.quadratic_rows(c, n)
    assert rows == [0b100, 0, 0b1, 0, 0, 0]


def test_popcount_and_degree():
    xs = np.array([0, 1, 3, 2**32 - 1, 2**63 + 5], dtype=np.uint64)
    assert cube.popcount(xs).tolist() == [0, 1, 2, 32, 3]
    assert cube.degree(4).tolist() == [bin(m).count("1") for m in range(16)]


def test_refusals():
    with pytest.raises(ValueError, match="2-cube"):
        cube.mobius([1, 2, 3], 2)
    with pytest.raises(ValueError, match="not in 1..64"):
        cube.mobius([1, 2], 1, bits=65)


def test_quadratic_rows_plane():
    n = 3
    c = np.zeros(1 << n, dtype=np.int64)
    c[0b011] = 2
    c[0b110] = 3
    assert cube.quadratic_rows(c, n, plane=0) == [0, 0b100, 0b010]
    assert cube.quadratic_rows(c, n, plane=1) == [0b010, 0b101, 0b010]
    for u, v in itertools.combinations(range(n), 2):
        assert (cube.quadratic_rows(c, n)[u] >> v & 1) == (cube.quadratic_rows(c, n)[v] >> u & 1)
