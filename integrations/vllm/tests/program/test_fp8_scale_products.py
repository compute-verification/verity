"""`kernels.fp8_twins.scale_products`: the scale products a host committer commits for `ScaledMmFp8BlockSharedScale_v1` are the circuit's
`F32Mul_v1(x_s[kb], w_s[nb][kb])` word for word -- the reference `evaluate` on every ordered pair of edge words where a host multiply can
differ from it (two NaNs of different payloads and signs, in both orders; NaN against finite, zero and infinity; signed zeros,
subnormals, +-max, +-inf; products that overflow, or underflow to a subnormal or to zero) -- and, fed to
`ScaledMmFp8BlockCoordinateGivenScale_v1`, give `ScaledMmFp8BlockCoordinate_v1`'s word."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import fp8
from verity.ml.definitions.numerics import fp8_twins
from verity.ml.definitions.registry import prims as P

NAN = [0x7FC00000, 0xFFC00000, 0x7FC00001, 0xFFC00001, 0x7FFFFFFF, 0xFFFFFFFF, 0x7F800001, 0xFF800001, 0x7FA00000, 0xFFBFFFFF]
INF = [0x7F800000, 0xFF800000]
FINITE = [0x00000000, 0x80000000,                                   # +-0
          0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00400000,  # subnormals
          0x00800000, 0x80800000, 0x3F800000, 0xBF800000,            # smallest normal, +-1
          0x7F7FFFFF, 0xFF7FFFFF,                                    # +-max
          0x5F800000, 0xDF800000, 0x1F800000, 0x9F800000, 0x0DA24260,  # 2^64, 2^-64, 1e-30: products overflow / underflow
          0x3B124925, 0x2B8CBCCC]                                    # 1/448 and 1e-12, scale-sized
EDGE = NAN + INF + FINITE


def _ref(a: int, b: int) -> int:
    return evaluate(P.F32Mul, a, b)[0]


def _is_nan(u: int) -> bool:
    return (u & 0x7FFFFFFF) > 0x7F800000


def _class(u: int) -> str:
    e, m = u & 0x7F800000, u & 0x007FFFFF
    return "nan" if e == 0x7F800000 and m else "inf" if e == 0x7F800000 else "zero" if not e and not m else "subnormal" if not e else "normal"


def test_every_ordered_edge_pair_is_f32mul():
    xs = np.asarray(EDGE, dtype=np.uint32)                       # x_s[kb] = EDGE[kb]
    ws = np.repeat(xs[:, None], len(EDGE), axis=1)               # w_s[nb][kb] = EDGE[nb]: every ordered pair (x_s, w_s) once
    got = fp8_twins.scale_products(xs, ws)
    assert got.shape == (len(EDGE), len(EDGE)) and got.dtype == np.uint32
    bad = [(hex(a), hex(b), hex(int(got[nb, kb])), hex(_ref(a, b))) for nb, b in enumerate(EDGE) for kb, a in enumerate(EDGE)
           if int(got[nb, kb]) != _ref(a, b)]
    assert bad == [], bad[:8]


def test_the_pairs_reach_every_case_and_the_operand_order_matters():
    ref = {(a, b): _ref(a, b) for a in EDGE for b in EDGE}
    two_nan = [(a, b) for a, b in ref if _is_nan(a) and _is_nan(b) and a != b]
    assert any((a ^ b) & 0x80000000 for a, b in two_nan) and any((a ^ b) & 0x7FFFFFFF for a, b in two_nan)   # signs and payloads differ
    assert any(ref[a, b] != ref[b, a] for a, b in two_nan)       # which NaN F32Mul_v1 returns depends on the order: (x_s, w_s), never swapped
    for a in NAN:                                                # NaN against finite, zero and infinity: a NaN
        assert all(_is_nan(ref[a, b]) and _is_nan(ref[b, a]) for b in FINITE + INF)
    fin = {_class(ref[a, b]) for a in FINITE for b in FINITE}
    assert fin == {"zero", "subnormal", "normal", "inf"}, fin     # overflow to inf, underflow to a subnormal and to zero
    assert _is_nan(ref[0x7F800000, 0x00000000]) and _is_nan(ref[0x80000000, 0xFF800000])   # inf x 0


def test_rows_and_shapes():
    rng = random.Random(0x5CA1E)
    ws = np.asarray([[rng.choice(EDGE) for _ in range(3)] for _ in range(4)], dtype=np.uint32)
    xs = np.asarray([[rng.choice(EDGE) for _ in range(3)] for _ in range(5)], dtype=np.uint32)
    got = fp8_twins.scale_products(xs, ws)
    assert got.shape == (5, 4, 3)
    for r in range(5):
        assert np.array_equal(got[r], fp8_twins.scale_products(xs[r], ws))
    for bad in ((xs[0, :2], ws), (xs[0], ws[0]), (xs[None], ws)):
        with pytest.raises(ValueError):
            fp8_twins.scale_products(*bad)


def test_the_products_feed_the_new_coordinate_to_the_old_word():
    K, G = 256, 128
    old, new = bind(fp8.ScaledMmFp8BlockCoordinate, K=K, G=G), bind(fp8.ScaledMmFp8BlockCoordinateGivenScale, K=K, G=G)
    rng = random.Random(0xF8B5)
    for k in range(24):
        xq = [rng.getrandbits(8) if rng.random() > 0.01 else 0x7F for _ in range(K)]
        wrow = [rng.getrandbits(8) for _ in range(K)]
        pool = EDGE if k % 3 == 2 else [rng.randrange(100, 127) << 23 | rng.getrandbits(23) for _ in range(8)]
        sx, sw = [rng.choice(pool) for _ in range(K // G)], [rng.choice(pool) for _ in range(K // G)]
        s = fp8_twins.scale_products(np.asarray(sx, dtype=np.uint32), np.asarray([sw], dtype=np.uint32))[0]
        assert evaluate(new, xq, [int(v) for v in s], wrow) == evaluate(old, xq, sx, wrow, sw), (k, [hex(v) for v in sx + sw])
