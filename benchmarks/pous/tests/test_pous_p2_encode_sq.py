"""P2's encode on the decode's squaring (`p2_v1/p2enc.cuh`, `encode_sq_kernel`) on CPU: a host build of the kernel, its
32 lanes in lockstep (`p2enc_twin.cpp`), gives the reference's bytes at the deployment's width and exponent, and its steps
around the squaring (the sums, differences and halvings mod p that make a product of two squarings, the root's choice,
the flags) agree with whole-number arithmetic on operands chosen to carry across every lane. The squaring itself is PTX
and is checked on a GPU by the decode's gate; here it is the product of whole numbers."""
from __future__ import annotations

import ctypes
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("verity")

BENCH = Path(__file__).resolve().parents[1]
CXX = next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)
pytestmark = pytest.mark.skipif(CXX is None, reason="needs a C++ compiler")

G_NONE, G_SAVE, G_BASE, G_MUL_BASE, G_MUL_SAVED, G_SWAP, G_MUL_END, G_ROOT = range(8)


def _p2():
    sys.path.insert(0, str(BENCH))
    from p2_v1 import gpu, reference
    return reference, gpu


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    lib = tmp_path_factory.mktemp("p2enc") / "libp2enc_twin.so"
    subprocess.run([CXX, "-O2", "-std=c++17", "-shared", "-fPIC", "-DP2SQ_KARA", "-I", str(BENCH / "p2_v1"),
                    str(BENCH / "tests" / "p2enc_twin.cpp"), "-o", str(lib)], check=True)
    so = ctypes.CDLL(str(lib))
    v, u = ctypes.c_void_p, ctypes.c_uint32
    so.twin_encode.argtypes = [v, v, v, v, u, u, u, u, u, v, v]
    so.twin_chain.argtypes = [u, u, u, v]
    return so


def _chain(twin, a: int, n: int, r: int) -> list[tuple[int, int]] | None:
    steps = (ctypes.c_uint32 * twin.twin_chain_max())()
    count = twin.twin_chain(a, n, r, steps)
    return None if count < 0 else [(s & 15, s >> 4) for s in steps[:count]]


def _encode(twin, P, split: tuple[int, int, int], blocks: list[tuple[int, int, int]]) -> tuple[list[int], int, int]:
    """The kernel's codewords of `(m, K, t)` blocks under the exponent `split`, its flags, and its squarings."""
    w, n = P.width_bytes, len(blocks)
    rows = [ctypes.create_string_buffer(b"".join(b[i].to_bytes(w, "big") for b in blocks), w * n) for i in range(3)]
    out = ctypes.create_string_buffer(b"\xa5" * (w * n), w * n)
    flags, squarings = ctypes.c_uint32(0), ctypes.c_uint64(0)
    assert twin.twin_encode(rows[0], out, rows[1], rows[2], n, P.c0, *split, ctypes.byref(flags), ctypes.byref(squarings)) == 0
    return [int.from_bytes(out.raw[i * w:(i + 1) * w], "big") for i in range(n)], flags.value, squarings.value


def _power_root(p: int, y: int, e: int) -> tuple[int, bool]:
    """What the kernel makes of `s = y^e`, whatever `e`: `s` with the parity `s² = ±y` asks for, and whether it is a root."""
    s = pow(y, e, p)
    even = s * s % p == y
    x = (p - s) % p if (s & 1) == even else s
    return x, even or (p - s * s) % p == y


def _model(P, e: int, m: int, k: int, t: int) -> tuple[int, bool]:
    r, ok1 = _power_root(P.p, (m + t) % P.p, e)
    c, ok2 = _power_root(P.p, r ^ k if r ^ k < P.p else r, e)
    return c, ok1 and ok2


def _operands(P, rng: random.Random) -> list[int]:
    """Values below p whose sums, differences and halves carry or borrow through whole lanes: the ends of the range, powers
    of two and their neighbours at lane boundaries, runs of ones and zeros, and uniform ones."""
    p, w = P.p, P.w
    edges = [0, 1, 2, 3, p - 1, p - 2, p - 3, p >> 1, (p >> 1) + 1, P.c0, P.c0 - 1, (1 << w - 1) - 1, 1 << w - 1]
    for bit in (32, 511, 512, 8192, 15872, 16384, 16416, w - 1):       # a limb's, a lane's and the two top limbs' first bits
        edges += [(1 << bit) - 1, 1 << bit, (1 << bit) + 1, p - (1 << bit), p - (1 << bit) - 1]
    runs = []
    for _ in range(6):
        lo, hi = sorted(rng.sample(range(w), 2))
        ones = (1 << hi) - (1 << lo)
        runs += [ones, p - 1 - ones]
    return edges + runs + [rng.randrange(p) for _ in range(6)]


@pytest.mark.parametrize("split", [(13, 16433, 2926), (0, 1, 0), (0, 1, 1), (3, 1, 0), (0, 2, 0), (1, 3, 5), (2, 13, 22), (5, 255, 1),
                                   (0, 256, 0xFFFF)])
def test_the_steps_are_the_exponent_with_two_squarings_a_product(twin, split):
    a, n, r = split
    e = (((1 << n) - 1) << a) + r
    q = (1 << 127) - 1                                           # any odd modulus: the steps only add, subtract, square and halve
    quarter = pow(4, -1, q)
    y = 0x1234_5678_9ABC_DEF0_0FED_CBA9_8765_4321 % q
    x, saved, scratch, squarings, products, ended = y, None, None, 0, 0, False
    for glue, count in _chain(twin, a, n, r):
        assert not ended
        if glue == G_SAVE:
            saved = x
        elif glue == G_BASE:
            x = y
        elif glue in (G_MUL_BASE, G_MUL_SAVED):
            b = y if glue == G_MUL_BASE else saved
            x, scratch = (x + b) % q, (x - b) % q
            assert count == 1
        elif glue == G_SWAP:
            x, scratch = scratch, x
            assert count == 1
        elif glue == G_MUL_END:
            x, products = (scratch - x) * quarter % q, products + 1
        elif glue == G_ROOT:
            assert (saved, x, count) == (pow(y, e, q), pow(y, 2 * e, q), 0)
            ended = True
        else:
            assert glue == G_NONE
        for _ in range(count):
            x, squarings = x * x % q, squarings + 1
    assert ended
    # a product by the saved power for each doubling, one by y for each further one of n and of r, one to join the two
    assert products == n.bit_length() - 1 + bin(n).count("1") - 1 + bin(r).count("1")
    assert squarings == n - 1 + a + (r.bit_length() - 1 if r else 0) + 1 + 2 * products


def test_the_deployments_root_is_16507_squarings_and_no_steps_for_no_ones(twin):
    R, gpu = _p2()
    split = gpu.exponent_split(R.DEPLOYMENT)
    a, n, r = split
    assert (((1 << n) - 1) << a) + r == (R.DEPLOYMENT.p + 1) // 4
    assert sum(count for _, count in _chain(twin, *split)) == 16_507
    assert _chain(twin, 3, 0, 5) is None


def test_the_kernels_arithmetic_around_the_squaring_on_operands_that_carry_across_lanes(twin):
    R, _ = _p2()
    P = R.DEPLOYMENT
    rng = random.Random(0x263)
    values = _operands(P, rng)
    for split in ((1, 3, 5), (0, 2, 3)):
        a, n, r = split
        e = (((1 << n) - 1) << a) + r
        blocks = []
        for i, (m, t) in enumerate(zip(values, rng.sample(values, len(values)), strict=True)):
            blocks += [(m, rng.choice(values), t), (m, rng.getrandbits(P.w), t)]
            if i % 8 == 0:                                               # a mask sigma leaves alone: r xor K is not below p
                blocks.append((m, _power_root(P.p, (m + t) % P.p, e)[0] ^ (P.p + i), t))
        got, flags, _ = _encode(twin, P, split, blocks)
        want = [_model(P, e, *b) for b in blocks]
        assert got == [c for c, _ in want]
        assert flags == (0 if all(ok for _, ok in want) else 2)


def test_a_tweak_not_below_p_and_a_power_that_is_no_root_raise_their_flags(twin):
    R, _ = _p2()
    P = R.DEPLOYMENT
    rng = random.Random(7)
    m, k = rng.getrandbits(8 * P.payload_bytes), rng.getrandbits(P.w)
    assert _encode(twin, P, (0, 1, 0), [(1, 0, 0)])[1] == 0              # 1 and p - 1 are first powers whose squares are +-them
    assert _encode(twin, P, (0, 1, 0), [(m, k, 5)])[1] == 2
    for t in (P.p, P.p + 3, (1 << P.w) - 1):
        got, flags, _ = _encode(twin, P, (0, 1, 0), [(m, k, t)])
        assert flags == 3 and got == [_model(P, 1, m, k, t - P.p)[0]]


def _root(P, v: int) -> tuple[int, tuple[str, ...]]:
    """The parity-canonical root of ±v from the kernel's power `s = v^((p + 1) / 4)`, and which way the kernel gets there.
    The power is `w − 2` squarings, each folded at `2^w = c0`, over `v^((c0 − 1) / 4)`."""
    if v == 0:
        return 0, ("zero",)
    s, mask = v, (1 << P.w) - 1
    for _ in range(P.w - 2):
        s *= s
        s = (s & mask) + P.c0 * (s >> P.w)
        s = (s & mask) + P.c0 * (s >> P.w)
    s = s * pow(pow(v, (P.c0 - 1) // 4, P.p), -1, P.p) % P.p
    of = "of y" if s * s % P.p == v else "of -y"
    r = s if (s & 1) == (of == "of -y") else P.p - s
    return r, (of, "kept" if r == s else "negated")


@pytest.mark.slow
def test_the_deployments_blocks_are_the_references_bytes(twin):
    """At the deployment's exponent, 33,014 squarings a block in the twin beside the reference's own roots: the whole chain
    against `reference.encode`, on roots of y and of -y, each kept and negated, on y = 0 and on a mask sigma leaves alone.
    The short exponents above can't hold the kernel to the reference."""
    R, gpu = _p2()
    P = R.DEPLOYMENT
    rng = random.Random(0x16448)
    draws = [(rng.getrandbits(8 * P.payload_bytes), rng.getrandbits(P.w), rng.randrange(P.p)) for _ in range(2)]
    t = rng.randrange(1, P.p)
    draws.append((P.p - t, rng.getrandbits(P.w - 1), t))                 # y = 0, and the second root is K's
    draws.append((rng.getrandbits(8 * P.payload_bytes), None, rng.randrange(P.p)))
    blocks, cases = [], set()
    for m, k, t in draws:
        r, case = _root(P, (m + t) % P.p)
        k = r ^ (P.p + 7) if k is None else k                            # r xor K is not below p
        z = r ^ k if r ^ k < P.p else r
        cases |= {case, _root(P, z)[1], "sigma is the identity" if z == r else "sigma masks"}
        blocks.append((m, k, t))
    assert cases == {("zero",), ("of y", "kept"), ("of y", "negated"), ("of -y", "kept"), ("of -y", "negated"),
                     "sigma is the identity", "sigma masks"}
    got, flags, squarings = _encode(twin, P, gpu.exponent_split(P), blocks)
    assert got == [R.encode(P, *b) for b in blocks]
    assert flags == 0 and squarings == len(blocks) * 2 * 16_507
