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


def _matrix(rows, n):
    return [[(r >> j) & 1 for j in range(n)] for r in rows]


def _sympy_rank(rows, n):
    return DomainMatrix.from_list(_matrix(rows, n), GF(2)).rank() if rows else 0


@pytest.mark.parametrize("seed", range(40))
def test_rank_against_brute_force_and_sympy(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 9)
    rows = _random_rows(rng, rng.randint(0, 10), n, rng.choice([None, 1, 2, 3]))
    r = f2.rank(rows)
    assert 1 << r == len(_span_brute(rows))
    assert r == _sympy_rank(rows, n)


def test_from_matrix_reads_each_row_mod_2():
    rng = np.random.default_rng(3)
    for n in (1, 5, 64, 70):
        M = rng.integers(0, 2, size=(6, n))
        rows = f2.from_matrix(M)
        assert _matrix(rows, n) == M.tolist()
        assert f2.from_matrix(M + 2) == rows
    with pytest.raises(ValueError, match="two axes"):
        f2.from_matrix([1, 0, 1])
