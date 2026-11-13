"""`dsa_ds_mla`'s cache write on vectors worked by hand: the floor at FLT_MIN (a tile of zeros or NaNs, an amax whose product with
f32(1/448) is subnormal, lands on FLT_MIN, or is the next f32 above it), the largest finite value at 448, e4m3's ties to even and
its subnormals, a NaN and an infinity in a tile, -0; and the row's latent part, `DsaLatentRow_v1`'s words."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.definitions.registry import dsa_ds_mla as M
from verity.definitions.registry import dsa_norm as DN

T = 32
NEG0, PINF, NINF, QNAN = 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000
#: 448 * 2^-126: its product with f32(1/448) = 0x3B124925 is 2^-126 (1 + 4.47e-8), within half an ulp of FLT_MIN
AT_FLOOR = 0x04E00000


def f(v: float) -> int:
    return int(np.float32(v).view(np.uint32))


def tile(*head: int, fill: int = 0) -> tuple[list[int], int]:
    """DsMlaFp8Tile_v1{T=32} of ``head`` then ``fill``: (its 32 bytes, its scale word)."""
    out = evaluate(bind(M.DsMlaFp8Tile, T=T), [*head, *[fill] * (T - len(head))])
    return [int(v) for v in out[:T]], int(out[T])


def test_a_tile_of_zeros_takes_the_floor_and_keeps_each_sign():
    q, s = tile(*[0, NEG0] * (T // 2))
    assert s == M.F_FLT_MIN and q == [0x00, 0x80] * (T // 2)


def test_a_tile_of_nans_takes_the_floor_and_each_byte_is_nan():
    q, s = tile(fill=QNAN)
    assert s == M.F_FLT_MIN and q == [0x7F] * T


def test_448_has_scale_one_and_its_byte_is_the_largest_finite():
    """448 * f32(1/448) = 1 + 4.47e-8 rounds to 1.0; 448 is e4m3 1.75 * 2^8 (0x7E), 1.0 is 0x38, -448 is 0xFE."""
    q, s = tile(f(448.0), f(-448.0), fill=f(1.0))
    assert s == 0x3F800000 and q == [0x7E, 0xFE] + [0x38] * (T - 2)


def test_e4m3_rounds_ties_to_even_and_keeps_its_subnormals():
    """At scale 1: 17 and 19 are ties between 16, 18 and 20 (0x58, 0x5A); 2^-8 is twice e4m3's least subnormal 2^-9 (0x02),
    2^-10 half of it (a tie to 0), -2^-10 a tie to -0, 1.5 * 2^-9 a tie to 2^-8."""
    q, s = tile(f(448.0), f(17.0), f(19.0), f(2.0 ** -8), f(2.0 ** -10), f(-2.0 ** -10), f(1.5 * 2.0 ** -9))
    assert s == 0x3F800000 and q[:7] == [0x7E, 0x58, 0x5A, 0x02, 0x00, 0x80, 0x02]


def test_a_nan_is_dropped_by_the_max_and_its_byte_is_nan():
    """amax over (NaN, 3, ...) is 3; 3 / RN(3 * f32(1/448)) is just under 448, so 0x7E."""
    q, s = tile(QNAN, 0xFFC00000, fill=f(3.0))
    assert s == f(np.float32(3.0) * np.uint32(M.F_INV448).view(np.float32))
    assert q == [0x7F, 0x7F] + [0x7E] * (T - 2)


def test_an_infinity_makes_the_scale_infinite():
    """div.full(x, inf): x * 0.25 times rcp(inf) = 0, so each finite value is a zero of its sign and inf / inf is NaN."""
    q, s = tile(PINF, NINF, f(-2.0), fill=f(1.0))
    assert s == PINF and q == [0x7F, 0x7F, 0x80] + [0x00] * (T - 3)


@pytest.mark.parametrize("amax,scale,byte", [
    (AT_FLOOR - 1, M.F_FLT_MIN, 0x7E),      # the product 2^-126 (1 - 7.45e-8) rounds to 0x007FFFFF, a subnormal: the floor
    (AT_FLOOR, M.F_FLT_MIN, 0x7E),          # the product rounds to FLT_MIN itself
    (AT_FLOOR + 1, 0x00800001, 0x7E),       # 2^-126 (1 + 1.64e-7) rounds to the next f32 above FLT_MIN, the scale
    (f(2.0 ** -120), M.F_FLT_MIN, 0x68),    # 2^-120 / 448 is subnormal: the floor, and 2^-120 / 2^-126 = 64 (0x68)
])
def test_the_floor_holds_up_to_flt_min_and_no_further(amax, scale, byte):
    q, s = tile(amax, amax | NEG0, fill=0)
    assert s == scale and q[:2] == [byte, byte | 0x80] and q[2:] == [0] * (T - 2)


def test_the_rows_latent_is_dsa_latent_row_and_a_zero_row_takes_the_floor():
    """At the small binding: a zero row's r is rsqrt(eps), so its y are zeros, every scale the floor; a random row's latent is
    `DsaLatentRow_v1`'s."""
    KVL, R, TILE, EPS = 256, 8, 64, 1e-5
    row = bind(M.DsaLatentRowDsMla, KVL=KVL, R=R, TILE=TILE, EPS=EPS)
    latent = bind(DN.DsaLatentRow, KVL=KVL, R=R, EPS=EPS)
    rng = np.random.default_rng(0)
    w = ((1 + 0.1 * rng.standard_normal(KVL)).astype(np.float32).view(np.uint32) >> 16).tolist()
    cs = (rng.standard_normal(R).astype(np.float32).view(np.uint32) >> 16).tolist()
    zero = [0x8000 * (i % 2) for i in range(KVL)] + [0] * R
    out = [int(v) for v in evaluate(row, zero, w, cs)]
    assert out[KVL + R:KVL + R + KVL] == [0x80 * (i % 2) for i in range(KVL)] and out[-KVL // TILE:] == [M.F_FLT_MIN] * (KVL // TILE)
    kv = (rng.standard_normal(KVL + R).astype(np.float32).view(np.uint32) >> 16).tolist()
    assert [int(v) for v in evaluate(row, kv, w, cs)][:KVL + R] == [int(v) for v in evaluate(latent, kv, w, cs)]
