import itertools
import math
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
    assert cnf.count(xs) == _brute_count(nv, clauses)
    m = cnf.solve()
    assert (m is None) == (cnf.count(xs) == 0)
    if m:
        assert all(any(m[abs(l)] == (l > 0) for l in c) for c in clauses)


@pytest.mark.parametrize("k", [3, 4])
def test_word_gates_against_integer_arithmetic(k):
    for a in (1, 3, 5, 6):
        cnf = Cnf()
        x, y = cnf.word("x", k), cnf.word("y", k)
        s, t, p = cnf.add_words(x, y), cnf.xor_words(x, y), cnf.mul_const(x, a)
        out = cnf.word("out", 3 * k)
        cnf.equal(out, s + t + p)
        models = cnf.models(x + y + out)
        assert len(models) == 1 << (2 * k)
        M = (1 << k) - 1
        for m in models:
            xv, yv, ov = cnf.value(m, x), cnf.value(m, y), cnf.value(m, out)
            assert ov == ((xv + yv) & M) | ((xv ^ yv) << k) | (((a * xv) & M) << (2 * k))


def test_names_constants_equal_differ():
    cnf = Cnf()
    assert cnf.var("a") == cnf.var("a") != cnf.var("b")
    x = cnf.word("x", 4)
    cnf.differ(x, cnf.const(4, 9))
    assert sorted(cnf.value(m, x) for m in cnf.models(x)) == [v for v in range(16) if v != 9]
    cnf.equal(cnf.add_words(x, cnf.const(4, 3)), cnf.const(4, 1))
    (m,) = cnf.models(x)
    assert cnf.value(m, x) == 14
    with pytest.raises(ValueError, match="out of range"):
        cnf.add([cnf.top + 1])


@pytest.mark.parametrize("n,k", [(5, 0), (5, 2), (6, 3), (6, 6)])
def test_cardinality_and_parity(n, k):
    for how, want in (("at_most", sum(math.comb(n, i) for i in range(k + 1))),
                      ("at_least", sum(math.comb(n, i) for i in range(k, n + 1))), ("exactly", math.comb(n, k))):
        cnf = Cnf()
        xs = cnf.word("x", n)
        getattr(cnf, how)(xs, k)
        assert cnf.count(xs) == want, how
    for odd in (False, True):
        cnf = Cnf()
        xs = cnf.word("x", n)
        cnf.parity(xs, odd)
        got = {cnf.value(m, xs) for m in cnf.models(xs)}
        assert got == {v for v in range(1 << n) if bin(v).count("1") % 2 == odd}


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
