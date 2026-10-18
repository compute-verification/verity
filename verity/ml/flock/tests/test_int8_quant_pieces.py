"""C-Flock's pieces for the int8 row's quantizer, its s8 casts and its float64 A (`verity_vllm.program.registry.int8_exact`), and
the LUT softmax's `I32Sub_v1` (`verity_vllm.program.registry.dsa`).  Lowered by `ir_lower.lower` into a C-Flock circuit and
evaluated there, each agrees with its Definition's reference on uniform words and on its edges: ±0, subnormals, the largest
finite values, the infinities, quiet and signalling NaNs of both signs, every integer-boundary exponent with the fraction at 0,
1/2 and 1/2 ± ulp (round to nearest even: Rint, RniSat), the saturation edges (±127, ±128, ±129 for the s8 cast; 2^32 - 1
± 1/2 for A), and for the float64 multiply, products whose exponent lands on the subnormal boundary or on overflow, with
significands short enough for exact ties."""
from __future__ import annotations

import struct

import numpy as np

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity_flock import ir_lower as IL
from verity_flock import tail_pieces as TP
from verity_flock.gf2 import Circuit
from verity_vllm.program.registry import int8_exact as X
from verity_vllm.program.registry.dsa import I32Sub

U64 = np.uint64


def _agree(prim, *inputs: np.ndarray) -> np.ndarray:
    """The piece's outputs on ``inputs`` (one array a parameter, an array parameter's T x K), held to the Definition's reference;
    the reference."""
    params, _ = E.batch.signature_of(prim)
    widths = [w for pt in params for w in pt.leaf_widths()]
    C = Circuit()
    ins = [C.input(f"in{j}", w) for j, w in enumerate(widths)]
    (out,) = IL.lower(C, prim, ins)
    C.output("out", out)
    inputs = [np.asarray(x, dtype=U64) for x in inputs]
    leaves = [c for x in inputs for c in (x.T if x.ndim == 2 else [x])]
    T = len(leaves[0])
    got_s, fail = C.evaluate({f"in{j}": IL.slice_words(x, w) for j, (x, w) in enumerate(zip(leaves, widths))}, T)
    assert not fail
    got = IL.unslice_words(got_s["out"], T)
    ref = E.evaluate_batch(prim, *inputs).astype(U64).reshape(T, -1)[:, 0]
    bad = np.flatnonzero(got != ref)
    assert not bad.size, (f"{prim.id}: {bad.size} of {T} instances differ from the reference, first at inputs "
                          f"{[hex(int(x[bad[0]])) for x in leaves]}: {hex(int(got[bad[0]]))} != {hex(int(ref[bad[0]]))}")
    return ref


def _f32(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _f64(v: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", v))[0]


F32_SPECIAL = [0, 1 << 31, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000,
               0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FBFFFFF, 0x7FFFFFFF, 0xFFC00123]
F32_VALUES = [_f32(v) for v in (0.5, -0.5, 0.49999997, 0.50000006, 1.0, -1.0, 1.5, -1.5, 2.5, -2.5, 3.5, 126.5, 127.0, 127.4, 127.5,
                                127.99999, 128.0, -127.0, -127.99999, -128.0, -128.5, -128.99998, -129.0, 129.0, 255.5, 4194302.5,
                                4194303.5, 8388607.5, -8388607.5, 8388608.0, 8388609.0, 16777216.0)]
F64_SPECIAL = [0, 1 << 63, 1, (1 << 63) | 1, 0x000FFFFFFFFFFFFF, 0x0010000000000000, 0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF,
               0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000, 0xFFF8000000000000, 0x7FF0000000000001,
               0xFFF0000000000001, 0x7FF8000000000123, 0x7FF7FFFFFFFFFFFF]
F64_VALUES = [_f64(v) for v in (1.0, -1.0, 0.5, 2.0, 1.5, 3.0, 0.1, 1e308, 1e-308, 2.0 ** -1074, 2.0 ** -1022, 2.0 ** 1023,
                                1.4426950408889634, 16.0, 4294967296.0, 0.08838834764831845)]


def _f32_words(rng, n: int) -> np.ndarray:
    """The f32 edges, every integer-boundary exponent (120..154, both signs) with the fraction at 0, ulp, 1/2 - ulp, 1/2, 1/2 + ulp
    and 1 - ulp, then n words: a third with |x| near the integers' range (exponents 100..159), the rest uniform."""
    edges = F32_SPECIAL + F32_VALUES
    for e in range(120, 155):
        edges += [(s << 31) | (e << 23) | f for f in (0, 1, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFF) for s in (0, 1)]
    w = rng.integers(0, 1 << 32, n, dtype=U64)
    k = n // 3
    w[:k] = ((rng.integers(0, 2, k, dtype=U64) << U64(31)) | (rng.integers(100, 160, k, dtype=U64) << U64(23))
             | rng.integers(0, 1 << 23, k, dtype=U64))
    return np.concatenate([np.asarray(edges, dtype=U64), w])


def _f64_words(rng, n: int) -> np.ndarray:
    """n f64 words with a random sign; 8% zero or subnormal, the rest with a significand of 1..53 significant bits (odd, so that
    products of short ones tie) and an exponent uniform (30%) or in 400..1649."""
    out = []
    for _ in range(n):
        if rng.random() < 0.08:
            out.append(int(rng.integers(0, 1 << 52)) | (int(rng.integers(0, 2)) << 63))
            continue
        j = int(rng.integers(1, 54))
        sig = (int(rng.integers(1 << (j - 1), 1 << j)) | 1) << (53 - j)
        e = int(rng.integers(1, 2047)) if rng.random() < 0.3 else int(rng.integers(400, 1650))
        out.append((int(rng.integers(0, 2)) << 63) | (e << 52) | (sig & ((1 << 52) - 1)))
    return np.asarray(out, dtype=U64)


def test_the_i32_sub_piece_wraps() -> None:
    rng = np.random.default_rng(71)
    edges = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    a, b = (np.asarray(x, dtype=U64) for x in zip(*[(p, q) for p in edges for q in edges]))
    _agree(I32Sub, np.concatenate([a, rng.integers(0, 1 << 32, 2000, dtype=U64)]),
           np.concatenate([b, rng.integers(0, 1 << 32, 2000, dtype=U64)]))


def test_the_f32_rint_piece_rounds_to_nearest_even() -> None:
    """Ties go to even (0.5, 1.5, 2.5, 2^22 + 1/2, 2^23 - 1/2), 1/2 ± ulp off them; NaNs come out quiet with their payload."""
    w = _f32_words(np.random.default_rng(72), 4000)
    ref = _agree(X.F32Rint, w)
    at = {int(x): int(r) for x, r in zip(w, ref)}
    assert [at[_f32(v)] for v in (0.5, 1.5, 2.5, -2.5, 4194302.5, 8388607.5, 0.50000006)] == \
           [0, _f32(2.0), _f32(2.0), _f32(-2.0), _f32(4194302.0), _f32(8388608.0), _f32(1.0)]
    assert at[0x7F800001] == 0x7FC00001 and at[0xFFC00123] == 0xFFC00123


def test_the_f32_to_s8_rzi_piece_truncates_and_saturates() -> None:
    w = _f32_words(np.random.default_rng(73), 4000)
    ref = _agree(X.F32ToS8Rzi, w)
    at = {int(x): int(r) for x, r in zip(w, ref)}
    assert [at[_f32(v)] for v in (127.99999, 128.0, -128.99998, -129.0, -127.99999, 1.5, -1.5, 0.5)] == \
           [127, 127, 0x80, 0x80, 0x81, 1, 0xFF, 0]
    assert (at[0x7FC00000], at[0x7F800000], at[0xFF800000]) == (0, 0x7F, 0x80)


def test_the_s8_to_f32_piece_is_exact_on_every_code() -> None:
    _agree(X.S8ToF32, np.arange(256, dtype=U64))


def test_the_f32_mul_wide_piece_is_exact() -> None:
    """Every pair of f32 edges (inf x 0 the default NaN, an f32 NaN widened then quieted), and uniform pairs."""
    rng = np.random.default_rng(74)
    e = np.asarray(F32_SPECIAL + F32_VALUES, dtype=U64)
    pa, pb = np.meshgrid(e, e)
    _agree(X.F32MulWide, np.concatenate([pa.ravel(), _f32_words(rng, 3000)]),
           np.concatenate([pb.ravel(), rng.permutation(_f32_words(rng, 3000))]))


def test_the_f64_mul_piece_rounds_to_nearest_even() -> None:
    """Every pair of f64 edges; short significands whose products are exact or tie; uniform words; and products aimed at the
    subnormal boundary, deep underflow, f64's top binade and overflow."""
    rng = np.random.default_rng(75)
    e = np.asarray(F64_SPECIAL + F64_VALUES, dtype=U64)
    pa, pb = np.meshgrid(e, e)
    na, nb = _f64_words(rng, 2500), _f64_words(rng, 2500)
    want = rng.choice([1, 2, 2046, 2047, 1023 - 51, 1023 - 60, 1023 + 1022], 2500)
    eb = np.clip(want - ((na >> U64(52)) & U64(0x7FF)).astype(np.int64), 1, 2046).astype(U64)
    nb = (nb & ~(U64(0x7FF) << U64(52))) | (eb << U64(52))
    a = np.concatenate([pa.ravel(), _f64_words(rng, 4000), rng.integers(0, 1 << 63, 1000, dtype=U64) * U64(2), na])
    b = np.concatenate([pb.ravel(), _f64_words(rng, 4000), rng.integers(0, 1 << 63, 1000, dtype=U64) * U64(2) + U64(1), nb])
    _agree(X.F64Mul, a, b)


def test_the_f64_to_u32_rni_sat_piece_rounds_and_saturates() -> None:
    """2^k + {0, 1/4, 1/2, 3/4} and their neighbours for every k < 33 (ties to even), 2^32 - 2 + 1/2 and 2^32 - 1 + {0,
    1/2 - ulp, 1/2} (the saturation by rounding), ±0, negatives, a subnormal, ±inf, NaNs, and words with exponents around
    [1/2, 2^32)."""
    rng = np.random.default_rng(76)
    vals = [_f64(v) for v in (0.0, -0.0, 0.25, 0.5, 0.5000000000000001, 1.0, 1.5, 2.5, 3.5, 4294967294.5, 4294967295.0,
                              4294967295.4999995, 4294967295.5, 4294967296.0, 1e300, float("inf"), float("-inf"), -1.0, -0.5,
                              2.0 ** -1074, 2.0 ** 31, 2.0 ** 31 + 0.5)] + [0x7FF8000000000000, 0xFFF8000000000000, 0x7FF0000000000001]
    for k in range(33):
        for frac in (0.0, 0.25, 0.5, 0.75):
            v = 2.0 ** k + frac
            vals += [_f64(v), _f64(v) + 1, _f64(v) - 1, _f64(v - 1 + frac)]
    for _ in range(4000):
        e = int(rng.integers(1015, 1060)) if rng.random() < 0.8 else int(rng.integers(0, 2048))
        vals.append((int(rng.random() < 0.15) << 63) | (e << 52) | int(rng.integers(0, 1 << 52)))
    ref = _agree(X.F64ToU32RniSat, np.asarray(vals, dtype=U64))
    assert vals[11] == _f64(4294967295.5) - 1
    assert ref[[3, 6, 7, 8, 9, 11, 12, 13, 15, 16]].tolist() == [0, 2, 2, 4, 0xFFFFFFFE] + [0xFFFFFFFF] * 4 + [0]


def _nnz(prim) -> int:
    """The terms of every form the ANDs of ``prim``'s lowering read (``nnz_AB``)."""
    params, _ = E.batch.signature_of(prim)
    C = Circuit()
    ins = [C.input(f"in{j}", w) for j, w in enumerate(w for pt in params for w in pt.leaf_widths())]
    C.output("out", IL.lower(C, prim, ins)[0])
    return C.counts()["nnz_AB"]


def test_a_rows_running_max_stays_linear_in_its_length() -> None:
    """``Int8RowScale_v1``'s amax is ``m = fmaxf(m, |x_k|)`` along the row, and PTX max's result carries its first operand's form
    (5 terms more a step): the max piece commits a bit past ``TP.FORM_TERMS``, so the lowering's ``nnz_AB`` grows with K, not
    K^2 (unbounded, x65 from K = 16 to 128, and the row's K = 4096 lowering held 250 GiB); the scale agrees with the reference,
    NaN, ±inf, ±0 and subnormals in the row."""
    rng = np.random.default_rng(77)
    edges = np.asarray([0x7FC0, 0xFFC0, 0x7F81, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x0001, 0x807F, 0x7F7F, 0xFF7F, 0x3F80], dtype=U64)
    nnz = {}
    for K in (16, 128):
        prim = bind(X.Int8RowScale, K=K)
        rows = rng.integers(0, 1 << 16, size=(K, 400), dtype=U64)
        rows[:, :len(edges)] = edges[rng.integers(0, len(edges), size=(K, len(edges)))]
        rows[rng.integers(0, K, 60), rng.integers(0, 400, 60)] = rng.choice(edges, 60)
        _agree(prim, rows.T)
        nnz[K] = _nnz(prim)
    assert nnz[128] < 10 * nnz[16], nnz
