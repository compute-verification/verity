import numpy as np
import pytest
from exact_kit import search, wrap

N = 5
MASK = (1 << N) - 1


def test_meet_against_brute_force():
    rng = np.random.default_rng(5)
    for shape in [(200,), (150, 3)]:
        L = rng.integers(0, 6, size=shape)
        R = rng.integers(0, 6, size=(170,) + shape[1:])
        want = sorted((i, j) for i in range(len(L)) for j in range(len(R)) if np.array_equal(L[i], R[j]))
        assert search.meet(L, R) == want
    with pytest.raises(ValueError, match="different lengths"):
        search.meet(np.zeros((2, 2)), np.zeros((2, 3)))


def test_meet_kernel_against_brute_force():
    """(a, b) with rotl(y ^ a, 2) * 11 - rotl(y, 2) * 11 == b for every y: the multiplied XOR difference is constant."""
    xs = np.arange(1 << N, dtype=np.uint64)
    A = np.arange(1 << N, dtype=np.uint64)
    B = np.arange(1 << N, dtype=np.uint64)

    def g(y, a):
        return ((wrap.rotl(y ^ a, 2, N) * np.uint64(11)) - (wrap.rotl(y, 2, N) * np.uint64(11))) & np.uint64(MASK)

    def h(y, b):
        return np.broadcast_to(b, np.broadcast_shapes(np.shape(y), np.shape(b)))

    want = sorted((int(a), int(b)) for a in A for b in B if np.all(g(xs, a) == h(xs, b)))
    got = search.meet_kernel(g, h, xs, A, B, probe=3)
    assert got == want and (0, 0) in got
    assert all(a in (0, int(wrap.rotr(1 << (N - 1), 2, N))) for a, _ in got)
