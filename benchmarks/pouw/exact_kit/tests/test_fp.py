import itertools
import random

import pytest
from exact_kit import fp
from sympy import GF
from sympy.polys.matrices import DomainMatrix


def _sympy(rows, p):
    M = DomainMatrix.from_list([list(r) for r in rows], GF(p))
    basis = [[int(x) % p for x in row] for row in M.rref()[0].to_Matrix().tolist() if any(int(x) % p for x in row)]
    null = [[int(x) % p for x in row] for row in M.nullspace().to_Matrix().tolist()] if M.rank() < M.shape[1] else []
    return M.rank(), basis, null


@pytest.mark.parametrize("p,seed", [(p, s) for p in (2, 3, 5, 7) for s in range(12)])
def test_rank_rref_nullspace_against_sympy(p, seed):
    rng = random.Random(p * 1000 + seed)
    n, m = rng.randint(1, 6), rng.randint(1, 6)
    rows = [[rng.randrange(p) for _ in range(n)] for _ in range(m)]
    if rng.random() < 0.5:
        rows.append([(a + 2 * b) % p for a, b in zip(rows[0], rows[-1])])
    r, sb, sn = _sympy(rows, p)
    basis, pivots = fp.rref(rows, p)
    assert fp.rank(rows, p) == r == len(basis)
    assert basis == sb
    ns = fp.nullspace(rows, p, n)
    assert len(ns) == n - r
    assert all(sum(a * b for a, b in zip(row, v)) % p == 0 for row in rows for v in ns)
    assert fp.rank(ns + sn, p) == len(sn) == len(ns)


@pytest.mark.parametrize("p", (2, 3, 5))
def test_kernel_span_and_lines_against_brute_force(p):
    rng = random.Random(p)
    n = 4
    rows = [[rng.randrange(p) for _ in range(n)] for _ in range(2)]
    every = list(itertools.product(range(p), repeat=n))
    ker = {v for v in every if all(sum(a * b for a, b in zip(r, v)) % p == 0 for r in rows)}
    assert set(fp.kernel(rows, p, n)) == ker and len(list(fp.kernel(rows, p, n))) == len(ker)
    sp = {v for v in every if fp.rank(rows + [list(v)], p) == fp.rank(rows, p)}
    assert set(fp.span(rows, p)) == sp
    assert fp.lines(rows, p) == sorted({fp.normal(v, p) for v in sp if any(v)})
    assert len(fp.lines(rows, p)) == (len(sp) - 1) // (p - 1)


def test_normal_and_refusals():
    assert fp.normal([0, 2, 1], 3) == (0, 1, 2)
    assert fp.normal([0, 3, 1], 5) == (0, 1, 2)
    with pytest.raises(ValueError, match="zero vector"):
        fp.normal([0, 0], 3)
    with pytest.raises(ValueError, match="not prime"):
        fp.rank([[1]], 4)
    with pytest.raises(ValueError, match="different lengths"):
        fp.rref([[1, 0], [1]], 3)
    assert list(fp.kernel([[1, 0], [0, 1]], 3, 2)) == [(0, 0)]
