"""CPU invariants of the fast-decode prototypes through their twin (`algo.py`), which indexes the way `tcsq.cu` and
`nttsq.cu` do: the tensor-core tiling and fragments give z^2's coefficients exactly, dense and under each Karatsuba depth;
the Karatsuba plan is the one the kernel's constexpr digits, offsets and arena slots encode; the fold reduces every limb-sum
form the kernels produce to z^2 mod p within the bounds its 64-bit arithmetic needs; the NTT as the kernel orders it gives
the convolution; and the widths are the primes named. The device kernels are gated byte-exact against
`p2_v1.reference` at run time (`bench.py`), before anything is timed."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import algo as A  # noqa: E402
import model as M  # noqa: E402
from p2_v1 import reference as R  # noqa: E402

SMALL_PRIMES = [p for p in range(3, 2000) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def probable_prime(n: int) -> bool:
    return all(n % p for p in SMALL_PRIMES) and pow(2, n - 1, n) == 1


def conv(z: int, nbytes: int, radix_bytes: int = 1) -> np.ndarray:
    d = np.frombuffer(z.to_bytes(nbytes, "little"), dtype=f"<u{radix_bytes}").astype(np.int64)
    return np.convolve(d, d)


def rand_below(rng: random.Random, n: int) -> int:
    return rng.randrange(n)


@pytest.mark.parametrize("per", [pytest.param(16, marks=pytest.mark.slow), 8, 4, 2])
def test_widths_are_the_primes_named(per: int):
    w = A.Width(per)
    assert w.w == 32 * (32 * per + 2) and w.n == w.w // 8
    assert w.c0 % 4 == 1 and w.p % 4 == 3
    assert probable_prime(w.p)
    R.Params(w.w, w.c0, w.n - 2)


@pytest.mark.parametrize("per", [2, 4])
def test_narrow_widths_take_the_smallest_c0(per: int):
    w = A.Width(per)
    assert not any(probable_prime((1 << w.w) - c) for c in range(1, w.c0, 4))


def test_deployment_is_p2_v4():
    assert (A.DEPLOYMENT.w, A.DEPLOYMENT.c0, A.DEPLOYMENT.n - 2) == (R.DEPLOYMENT.w, R.DEPLOYMENT.c0, R.DEPLOYMENT.payload_bytes)
    assert A.DEPLOYMENT.q_dense == 65 and A.DEPLOYMENT.q_main == 64 and A.DEPLOYMENT.max_depth() == 3


@pytest.mark.parametrize("q", [1, 5, 9, 17])
def test_dense_tiles_square_exactly(q: int):
    rng = random.Random(q)
    for z in (rng.getrandbits(256 * q), (1 << 256 * q) - 1, 1):
        assert np.array_equal(A.tc_square(z, q)[:64 * q - 1], conv(z, 32 * q))


def test_dense_tiles_square_the_deployment_width():
    rng = random.Random(16448)
    q = A.DEPLOYMENT.q_dense
    for z in (rand_below(rng, A.DEPLOYMENT.p), A.DEPLOYMENT.p - 1):
        c = A.tc_square(z, q)
        assert np.array_equal(c[:2 * A.DEPLOYMENT.n - 1], conv(z, A.DEPLOYMENT.n)) and not c[2 * A.DEPLOYMENT.n - 1:].any()


def test_mma_counts():
    assert A.mma_count(65) == 1188 and A.mma_count(64) == 1152
    counts = [len(A.karatsuba_plan(64, d)) * A.mma_count(64 >> d) for d in range(4)]
    assert counts == [1152, 960, 864, 864]


@pytest.mark.parametrize("q,depth", [(16, 1), (32, 2), (64, 1), (64, 2), (64, 3)])
def test_karatsuba_squares_exactly(q: int, depth: int):
    """Karatsuba's coefficients are signed and differ from the schoolbook's (D's digits carry borrows); their value is z^2,
    and each fits the kernel's s32 accumulators."""
    rng = random.Random(100 * q + depth)
    for z in (rng.getrandbits(256 * q), (1 << 256 * q) - 1, (1 << 128 * q) - 1, rng.getrandbits(128 * q)):
        c, mmas = A.kara_square(z, q, depth)
        assert A.coeff_value(c) == z * z and np.abs(c).max() < 1 << 31
        assert mmas == len(A.karatsuba_plan(q, depth)) * A.mma_count(q >> depth)


def _cuda_plan(q: int, depth: int) -> list[tuple[str, tuple[int, ...], int]]:
    """tcsq.cu's constexpr `digit`, `noffs`, `sign` and `offset`, transcribed."""
    def digit(b: int, lvl: int) -> int:
        for _ in range(depth - lvl):
            b //= 3
        return b % 3

    def offset(b: int, k: int) -> int:
        o = 0
        for lvl in range(depth, 0, -1):
            t = (q >> lvl) // 8
            dg = digit(b, lvl)
            if dg == 2:
                o += t
            else:
                o += t if k & 1 else (0 if dg == 0 else 2 * t)
                k >>= 1
        return o

    out = []
    for b in range(3 ** depth):
        digits = "".join("01D"[digit(b, lvl)] for lvl in range(1, depth + 1))
        noffs = 2 ** sum(digit(b, lvl) != 2 for lvl in range(1, depth + 1))
        sign = (-1) ** sum(digit(b, lvl) == 2 for lvl in range(1, depth + 1))
        out.append((digits, tuple(offset(b, k) for k in range(noffs)), sign))
    return out


@pytest.mark.parametrize("per,depth", [(16, 1), (16, 2), (8, 1), (8, 2), (4, 1)])
def test_kernel_plan_is_the_twins(per: int, depth: int):
    q = A.Width(per).q_main
    assert _cuda_plan(q, depth) == list(A.karatsuba_plan(q, depth))


def test_kernel_arena_slots_follow_the_plan():
    # tcsq.cu's `prepare` at depth 2 writes slots 0 .. 8 as z00, z01, |z00 - z01|, z10, z11, |z10 - z11|, the halves of
    # D1 = |z0 - z1| and |D10 - D11|
    assert [d for d, _, _ in A.karatsuba_plan(64, 2)] == ["00", "01", "0D", "10", "11", "1D", "D0", "D1", "DD"]
    rng = random.Random(2)
    z = rng.getrandbits(256 * 64)
    h = 1 << 8 * 1024
    z0, z1 = z % h, z // h
    d1 = abs(z0 - z1)
    q4 = 1 << 8 * 512
    want = [z0 % q4, z0 // q4, abs(z0 % q4 - z0 // q4), z1 % q4, z1 // q4, abs(z1 % q4 - z1 // q4), d1 % q4, d1 // q4,
            abs(d1 % q4 - d1 // q4)]
    assert A.kara_operands(z, 64, 2) == want


def _check_fold(v: list[int], width: A.Width, z: int) -> None:
    bounds: dict = {}
    assert A.fold(v, width, bounds) % width.p == z * z % width.p
    assert 0 < bounds["x_min"] and bounds["x_max"] < 1 << 63


def _operands(width: A.Width, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rand_below(rng, width.p) for _ in range(3)] + [0, 1, width.p - 1, (1 << width.w - 1) - 1]


@pytest.mark.parametrize("per", sorted(A.WIDTHS))
def test_fold_of_the_dense_square(per: int):
    width = A.Width(per)
    for z in _operands(width, per):
        _check_fold(A.limb_sums(conv(z, width.n), width), width, z)


@pytest.mark.parametrize("per,depth", [(16, 1), (16, 2), (8, 1), (8, 2), (4, 1)])
def test_fold_of_the_karatsuba_square_with_the_top_terms(per: int, depth: int):
    width = A.Width(per)
    lm = 32 * per
    for z in _operands(width, 7 * per + depth):
        main = z % (1 << 32 * lm)
        c, _ = A.kara_square(main, width.q_main, depth)
        v = A.limb_sums(c, width)
        e = A.top_terms(z, width)
        _check_fold([a + b for a, b in zip(v, e)], width, z)


def test_ntt_square_is_the_convolution():
    width = A.DEPLOYMENT
    for z in _operands(width, 2048)[:3] + [(1 << width.w) - 1]:
        c = A.ntt_square(z, width.n)
        assert c == conv(z, width.n, 2).tolist()
        assert max(c) <= 1028 * 0xFFFF ** 2 < A.GOLDILOCKS


def test_fold_of_the_ntt_square():
    width = A.DEPLOYMENT
    for z in _operands(width, 4096):
        _check_fold(A.limb_sums16(A.ntt_square(z, width.n), width), width, z)


def test_ntt_root_has_order_2048():
    w = A.ntt_root()
    assert pow(w, 2048, A.GOLDILOCKS) == 1 and pow(w, 1024, A.GOLDILOCKS) == A.GOLDILOCKS - 1


@pytest.mark.parametrize("alpha", [1.585, 1.404])
def test_seqroot_model_is_the_specs_at_16448(alpha: float):
    s = M.seqroot(16_448, alpha)
    assert s["root_call_ns"] == 854_029 and s["rtt_cap_us"] == 354 and round(s["root_us"], 1) == 5960.5


@pytest.mark.parametrize("w", [8_256, 4_160, 2_112])
def test_narrower_widths_leave_no_room_for_delta(w: int):
    for alpha in (1.585, 1.404):
        s = M.seqroot(w, alpha)
        assert s["rtt_cap_us"] is None and s["root_call_ns"] < 200_000


def test_route_ceilings():
    assert 75 < M.tc_ceiling_gbps(1188) < 76 and M.tc_ceiling_gbps(486) < 2 * M.MEASURED["keys_GBps"]
    assert M.full_pass_gbps(M.tc_ceiling_gbps(486)) < 0.07 * M.MEASURED["read_GBps"]
    assert round(M.partial_fraction(), 3) == 0.036


def test_decode_is_the_references():
    rng = random.Random(5)
    P = R.DEPLOYMENT
    for _ in range(4):
        c, k, t = rand_below(rng, P.p), rng.getrandbits(P.w), rand_below(rng, P.p)
        assert A.decode(A.DEPLOYMENT, c, k, t) == R.decode(P, c, k, t)
