import random

from verity_lin import code
from verity_lin.field import P


def test_ntt_evaluates_the_polynomial():
    rng = random.Random(3)
    coeffs = [rng.randrange(P) for _ in range(8)]
    n = 32
    w = code.root_of_unity(n)
    want = [sum(c * pow(w, i * j, P) for i, c in enumerate(coeffs)) % P for j in range(n)]
    assert code.encode(coeffs, n) == want


def test_the_code_is_linear_and_separates_rows():
    rng = random.Random(4)
    a = [rng.randrange(P) for _ in range(16)]
    b = [rng.randrange(P) for _ in range(16)]
    n = 64
    ea, eb = code.encode(a, n), code.encode(b, n)
    assert code.encode([(x + 3 * y) % P for x, y in zip(a, b)], n) == [(x + 3 * y) % P for x, y in zip(ea, eb)]
    b2 = list(a)
    b2[5] = (b2[5] + 1) % P
    assert sum(x != y for x, y in zip(ea, code.encode(b2, n))) >= code.distance(16, 4)
