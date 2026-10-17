import numpy as np
import pytest
from exact_kit import search, wrap

N = 5
MASK = (1 << N) - 1


def _step(p):
    """A 5-bit two-input map on packed points p = y | d << 5: rotate y, XOR d in, multiply."""
    y, d = p & np.uint64(MASK), p >> np.uint64(N)
    return ((wrap.rotl(y, 2, N) ^ d) * np.uint64(11)) & np.uint64(MASK)


def _brute_kernel(f, xs, deltas, act):
    base = f(xs)
    return [int(d) for d in deltas if np.array_equal(f(act(xs, np.uint64(d))), base)]


@pytest.mark.parametrize("probe", [1, 4, 64, 4096])
def test_kernel_against_brute_force(probe):
    xs = np.arange(1 << (2 * N), dtype=np.uint64)
    deltas = np.arange(1 << (2 * N), dtype=np.uint64)
    xor = lambda x, d: x ^ d
    want = _brute_kernel(_step, xs, deltas, xor)
    assert search.kernel(_step, xs, deltas, probe=probe) == want
    # rotl(y ^ e, 2) ^ (d ^ e') == rotl(y, 2) ^ d iff e' = rotl(e, 2): one kernel element per e
    assert len(want) == 1 << N and all((int(d) >> N) == int(wrap.rotl(int(d) & MASK, 2, N)) for d in want)


def test_kernel_with_addition_and_a_tuple_valued_map():
    M = (1 << 8) - 1
    xs = np.arange(1 << 8, dtype=np.uint64)
    deltas = np.arange(1 << 8, dtype=np.uint64)
    add = lambda x, d: (x + d) & np.uint64(M)
    f = lambda x: ((x * np.uint64(4)) & np.uint64(M), (x * np.uint64(8)) & np.uint64(M))
    assert search.kernel(f, xs, deltas, act=add) == _brute_kernel(lambda x: f(x)[0] * 256 + f(x)[1], xs, deltas, add)
    assert search.kernel(f, xs, deltas, act=add) == [0, 64, 128, 192]


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
    g = lambda y, a: ((wrap.rotl(y ^ a, 2, N) * np.uint64(11)) - (wrap.rotl(y, 2, N) * np.uint64(11))) & np.uint64(MASK)
    h = lambda y, b: np.broadcast_to(b, np.broadcast_shapes(np.shape(y), np.shape(b)))
    want = sorted((int(a), int(b)) for a in A for b in B if np.all(g(xs, a) == h(xs, b)))
    got = search.meet_kernel(g, h, xs, A, B, probe=3)
    assert got == want and (0, 0) in got
    assert all(a in (0, int(wrap.rotr(1 << (N - 1), 2, N))) for a, _ in got)


def test_lift_against_brute_force():
    for k in (1, 3, 8, 12):
        sq = search.lift(lambda c, j: (c * c - 1) % (1 << j) == 0, k)
        assert sq == [c for c in range(1 << k) if (c * c - 1) % (1 << k) == 0]
        inv = search.lift(lambda c, j: (c * 0x5B - 1) % (1 << j) == 0, k)
        assert inv == [pow(0x5B, -1, 1 << k)]
