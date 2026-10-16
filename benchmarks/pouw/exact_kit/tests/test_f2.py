import itertools
import random

import numpy as np
import pytest
from exact_kit import f2
from sympy import GF
from sympy.polys.matrices import DomainMatrix


def _random_rows(rng, m, n, rank_at_most=None):
    if rank_at_most is None:
        return [rng.getrandbits(n) for _ in range(m)]
    base = [rng.getrandbits(n) for _ in range(rank_at_most)]
    return [_xor(b for b in base if rng.random() < 0.5) for _ in range(m)]


def _xor(it):
    v = 0
    for x in it:
        v ^= x
    return v


def _span_brute(rows):
    out = {0}
    for r in rows:
        out |= {x ^ r for x in out}
    return out


def _sympy_rank(rows, n):
    M = f2.to_matrix(rows, n).astype(int).tolist()
    return DomainMatrix.from_list(M, GF(2)).rank() if rows else 0


@pytest.mark.parametrize("seed", range(40))
def test_rank_rref_span_against_brute_force_and_sympy(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 9)
    rows = _random_rows(rng, rng.randint(0, 10), n, rng.choice([None, 1, 2, 3]))
    sp = _span_brute(rows)
    r = f2.rank(rows)
    assert 1 << r == len(sp)
    assert r == _sympy_rank(rows, n)
    basis, pivots = f2.rref(rows)
    assert len(basis) == r and pivots == sorted(pivots)
    for p, b in zip(pivots, basis):
        assert (b & -b).bit_length() - 1 == p
        assert all(not (o >> p & 1) for o in basis if o != b)
    shuffled = list(sp)
    rng.shuffle(shuffled)
    assert f2.rref(shuffled) == (basis, pivots)
    assert sorted(f2.span(rows)) == sorted(sp)
    for v in range(1 << n):
        assert f2.in_span(v, rows) == (v in sp)


@pytest.mark.parametrize("seed", range(30))
def test_nullspace_and_kernel_against_brute_force(seed):
    rng = random.Random(100 + seed)
    n = rng.randint(1, 9)
    rows = _random_rows(rng, rng.randint(0, 8), n)
    ker = {x for x in range(1 << n) if all(f2.dot(r, x) == 0 for r in rows)}
    ns = f2.nullspace(rows, n)
    assert len(ns) == n - f2.rank(rows)
    assert set(f2.kernel(rows, n)) == ker
    assert len(list(f2.kernel(rows, n))) == len(ker)


def test_nullspace_refuses_a_row_wider_than_n():
    with pytest.raises(ValueError, match="column >= n"):
        f2.nullspace([0b100], 2)


def test_matrix_round_trip_and_transpose():
    rng = np.random.default_rng(3)
    for n in (1, 5, 64, 70):
        M = rng.integers(0, 2, size=(6, n))
        rows = f2.from_matrix(M)
        assert np.array_equal(f2.to_matrix(rows, n), M)
        assert f2.from_matrix(M + 2) == rows
        assert np.array_equal(f2.to_matrix(f2.transpose(rows, n), 6), M.T)


def test_alternating_and_wedge_are_the_bilinear_forms():
    n = 5
    for a, b in itertools.product(range(1, 1 << n), repeat=2):
        if (a * 7 + b) % 11:
            continue
        W = f2.wedge(n, a, b)
        for x, y in itertools.product(range(1 << n), repeat=2):
            want = (f2.dot(a, x) * f2.dot(b, y) + f2.dot(a, y) * f2.dot(b, x)) % 2
            got = sum(((W[u] >> v) & 1) * (x >> u & 1) * (y >> v & 1) for u in range(n) for v in range(n)) % 2
            assert got == want
        assert all(not (W[u] >> u & 1) for u in range(n))
    A = f2.alternating(4, [(0, 1), (2, 3), (0, 1)])
    assert A == [0, 0, 0b1000, 0b0100]
    assert f2.xor_rows(A, f2.alternating(4, [(0, 1)])) == f2.alternating(4, [(0, 1), (2, 3)])
    assert f2.rank(f2.alternating(4, [(0, 2), (1, 3)])) == 4
    with pytest.raises(ValueError, match="diagonal"):
        f2.alternating(3, [(1, 1)])
