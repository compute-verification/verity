import itertools
import random

import pytest
from exact_kit.sat import Cnf


def _brute_count(nv, clauses):
    return sum(all(any((bits >> (abs(l) - 1) & 1) == (l > 0) for l in c) for c in clauses) for bits in range(1 << nv))


@pytest.mark.parametrize("seed", range(20))
def test_model_count_of_random_3cnf_against_brute_force(seed):
    rng = random.Random(seed)
    nv = 9
    cnf = Cnf()
    xs = [cnf.var(f"x{i}") for i in range(nv)]
    clauses = [[rng.choice([1, -1]) * rng.choice(xs) for _ in range(3)] for _ in range(rng.randint(10, 40))]
    for c in clauses:
        cnf.add(c)
    count = len(cnf.models(xs))
    assert count == _brute_count(nv, clauses)
    m = cnf.solve()
    assert (m is None) == (count == 0)
    if m:
        assert all(any(m[abs(l)] == (l > 0) for l in c) for c in clauses)


def test_names_and_the_constant():
    cnf = Cnf()
    assert cnf.var("a") == cnf.var("a") != cnf.var("b")
    x = cnf.word("x", 4)
    cnf.add([-b if 9 >> i & 1 else b for i, b in enumerate(x)])
    assert sorted(cnf.value(m, x) for m in cnf.models(x)) == [v for v in range(16) if v != 9]
    cnf.add([-cnf.true, x[0]])
    assert all(cnf.value(m, x) & 1 for m in cnf.models(x))
    with pytest.raises(ValueError, match="out of range"):
        cnf.add([cnf.top + 1])


@pytest.mark.parametrize("n", [5, 6])
def test_parity(n):
    for odd in (False, True):
        cnf = Cnf()
        xs = cnf.word("x", n)
        cnf.parity(xs, odd)
        got = {cnf.value(m, xs) for m in cnf.models(xs)}
        assert got == {v for v in range(1 << n) if bin(v).count("1") % 2 == odd}
    cnf = Cnf()
    cnf.parity([], True)
    assert cnf.solve() is None


def test_models_limit_and_unsat():
    cnf = Cnf()
    xs = cnf.word("x", 3)
    assert len(cnf.models(xs, limit=5)) == 5
    cnf.add([xs[0]])
    cnf.add([-xs[0]])
    assert cnf.solve() is None and cnf.models(xs) == []
    for a, b in itertools.product((0, 1), repeat=2):
        c = Cnf()
        u, v = c.var(), c.var()
        z = c.and_(u, v), c.or_(u, v), c.xor(u, v)
        c.add([u if a else -u])
        c.add([v if b else -v])
        m = c.solve()
        assert [m[abs(l)] == (l > 0) for l in z] == [bool(a and b), bool(a or b), bool(a ^ b)]
