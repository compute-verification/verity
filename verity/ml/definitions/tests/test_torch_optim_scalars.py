"""`host_f64` and `torch_optim`'s scalar Definitions: the f64 builders equal numpy's IEEE arithmetic (and an exact rational fma
rounded once) on random and special words; `pow_main` equals a Python twin of glibc 2.39's ``__pow_fma`` main path (each
``vfmadd`` an exact rational fma, each SSE2 operation CPython's float), and the twin equals glibc on pinned vectors measured on
an x86-64 glibc 2.39 FMA host and, on such a host, on random inputs; the bias corrections, ``** 0.5``, the seven f32 scalars and
SGD's ``f32(-lr)`` equal `_multi_tensor_adam`'s and `_multi_tensor_sgd`'s host code (with the twin's ``**``), and on such a
host the scalars torch 2.14 itself hands its foreach kernels."""
from __future__ import annotations

import math
import os
import random
import struct
from fractions import Fraction

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import glibc_pow_data as D, host_f64 as H, torch_optim as TO

U32, U64 = np.uint32, np.uint64
M64 = (1 << 64) - 1
F32, F64 = Array(32, BL.BIT), Array(64, BL.BIT)
FLT_MAX = float(np.finfo(np.float32).max)
TINY = 2.0 ** -1022


def _b(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _d(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(u) & M64))[0]


def _fma(a: float, b: float, c: float) -> float:
    """``RN(a * b + c)``, one rounding, for finite operands (IEEE's zero signs)."""
    v = Fraction(a) * Fraction(b) + Fraction(c)
    if v == 0:
        if a == 0 or b == 0:
            return -0.0 if (math.copysign(1, a) * math.copysign(1, b) < 0 and math.copysign(1, c) < 0) else 0.0
        return 0.0
    return float(v)


def pow_twin(x: float, y: float) -> tuple[float, bool]:
    """``(pow(x, y), big)`` on glibc 2.39's ``__pow_fma`` main path for a positive normal x, `host_f64.pow_main`'s steps in
    Python: ``e_pow.c``'s ``log_inline`` and ``exp_inline`` with each contraction the FMA build makes an exact fma. ``big`` is
    ``|y log x| >= 512`` (then the value is NaN here)."""
    ix = _b(x)
    tmp = (ix - H.POW_OFF) & M64
    i = (tmp >> 45) & 127
    k = (tmp >> 52) - (4096 if tmp >> 63 else 0)
    z = _d(ix - ((tmp >> 52) << 52))
    invc, logc, logctail = _d(D.LOG_INVC[i]), _d(D.LOG_LOGC[i]), _d(D.LOG_LOGCTAIL[i])
    A = [_d(a) for a in D.LOG_POLY]
    r = _fma(z, invc, -1.0)
    t1 = _fma(float(k), _d(D.LN2HI), logc)
    t2 = t1 + r
    lo1 = _fma(float(k), _d(D.LN2LO), logctail)
    lo2 = t1 - t2 + r
    ar = A[0] * r
    ar2 = r * ar
    ar3 = r * ar2
    hi = t2 + ar2
    lo3 = _fma(ar, r, -ar2)
    lo4 = t2 - hi + ar2
    q = _fma(ar2, _fma(_fma(r, A[6], A[5]), ar2, _fma(r, A[4], A[3])), _fma(r, A[2], A[1]))
    lo = _fma(ar3, q, lo1 + lo2 + lo3 + lo4)
    lhi = hi + lo
    llo = hi - lhi + lo
    ehi = y * lhi
    elo = _fma(y, llo, _fma(y, lhi, -ehi))
    top = (_b(ehi) >> 52) & 0x7FF
    if top < 0x3C9:
        return 1.0 + ehi, False
    if top >= 0x408:
        return math.nan, True
    kd = _fma(ehi, _d(D.INVLN2N), _d(D.SHIFT))
    ki = _b(kd)
    kd -= _d(D.SHIFT)
    rr = _fma(kd, _d(D.NEGLN2LON), _fma(kd, _d(D.NEGLN2HIN), ehi)) + elo
    j = ki & 127
    tail, scale = _d(D.EXP_TAB[2 * j]), _d(D.EXP_TAB[2 * j + 1] + (ki << 45))
    C2, C3, C4, C5 = (_d(c) for c in D.EXP_POLY)
    r2 = rr * rr
    t = _fma(r2 * r2, _fma(rr, C5, C4), _fma(r2, _fma(rr, C3, C2), tail + rr))
    return _fma(scale, t, scale), False


def one_minus_pow(beta: float, y: float) -> float:
    """CPython's ``1 - beta ** y`` for 0 <= beta < 1 and y >= 1: the power is 0 at beta zero, below 2^-1022 at beta subnormal
    and below e^-512 where the twin leaves its main path, and 1 minus any of those is 1."""
    if abs(beta) < TINY:
        return 1.0
    p, big = pow_twin(beta, y)
    return 1.0 if big else 1.0 - p


def pow_half(x: float) -> float:
    """CPython's ``x ** 0.5`` for a positive normal x (``float_pow``'s ``1.0 ** y`` is 1)."""
    return 1.0 if x == 1.0 else pow_twin(x, 0.5)[0]


#: (x, y, x ** y) as hex floats, measured with CPython on x86-64 glibc 2.39 (FMA and AVX2, so `__pow_fma`)
PINNED = [
    ("0x1.ccccccccccccdp-1", "0x1.0000000000000p+0", "0x1.ccccccccccccdp-1"),
    ("0x1.ccccccccccccdp-1", "0x1.0000000000000p+1", "0x1.9eb851eb851ecp-1"),
    ("0x1.ccccccccccccdp-1", "0x1.8000000000000p+1", "0x1.753f7ced91688p-1"),
    ("0x1.ccccccccccccdp-1", "0x1.4000000000000p+3", "0x1.650bf60432fd9p-2"),
    ("0x1.ccccccccccccdp-1", "0x1.9000000000000p+6", "0x1.bda056ed49410p-16"),
    ("0x1.ccccccccccccdp-1", "0x1.f400000000000p+9", "0x1.fee7413dac341p-153"),
    ("0x1.ccccccccccccdp-1", "0x1.0000000000000p+12", "0x1.50b3c206e427dp-623"),
    ("0x1.ccccccccccccdp-1", "0x1.0000000000000p+16", "0x0.0p+0"),
    ("0x1.ccccccccccccdp-1", "0x1.0000000000000p+24", "0x0.0p+0"),
    ("0x1.e666666666666p-1", "0x1.0000000000000p+0", "0x1.e666666666666p-1"),
    ("0x1.e666666666666p-1", "0x1.0000000000000p+1", "0x1.ce147ae147ae1p-1"),
    ("0x1.e666666666666p-1", "0x1.8000000000000p+1", "0x1.b6f9db22d0e55p-1"),
    ("0x1.e666666666666p-1", "0x1.4000000000000p+3", "0x1.328da5e9df383p-1"),
    ("0x1.e666666666666p-1", "0x1.9000000000000p+6", "0x1.8401ff6058453p-8"),
    ("0x1.e666666666666p-1", "0x1.f400000000000p+9", "0x1.ffcb2f699313dp-75"),
    ("0x1.e666666666666p-1", "0x1.0000000000000p+12", "0x1.db9abd4b89881p-304"),
    ("0x1.e666666666666p-1", "0x1.0000000000000p+16", "0x0.0p+0"),
    ("0x1.e666666666666p-1", "0x1.0000000000000p+24", "0x0.0p+0"),
    ("0x1.ff7ced916872bp-1", "0x1.0000000000000p+0", "0x1.ff7ced916872bp-1"),
    ("0x1.ff7ced916872bp-1", "0x1.c000000000000p+2", "0x1.fc6d3e72229bcp-1"),
    ("0x1.ff7ced916872bp-1", "0x1.f400000000000p+9", "0x1.788526411ed72p-2"),
    ("0x1.ff7ced916872bp-1", "0x1.e241000000000p+16", "0x1.bdbf68d13d1a6p-179"),
    ("0x1.99999999999a0p-5", "0x1.0000000000000p-1", "0x1.c9f25c5bfedddp-3"),
    ("0x1.8f5c28f5c28f8p-4", "0x1.0000000000000p-1", "0x1.3fbe701157609p-2"),
    ("0x1.cf4bf0995ab00p-3", "0x1.0000000000000p-1", "0x1.e70a37edad9ffp-2"),
    ("0x1.d89aab0f30a25p-1", "0x1.0000000000000p-1", "0x1.ebe86917648e6p-1"),
    ("0x1.0000000000000p+0", "0x1.0000000000000p-1", "0x1.0000000000000p+0"),
    ("0x1.e7ea40a393788p-4", "0x1.ebac5fe3b9bc0p+4", "0x1.98c3213865672p-95"),
    ("0x1.3e332a239ea9bp-1", "0x1.6ab7ca3b9c6b0p+4", "0x1.5c2e6295ae6cap-16"),
    ("0x1.af4f47aec1269p-1", "0x1.41502f9bb10f8p+4", "0x1.059d516bb1548p-5"),
    ("0x1.c3ae681eb628ep-2", "0x1.4c40000000000p+12", "0x0.0p+0"),
    ("0x1.26224c688ba2cp-2", "-0x1.01c46af8a5ac0p+2", "0x1.301838511a3bbp+7"),
    ("0x1.f60db9d6f6378p-2", "0x1.0000000000000p-1", "0x1.66813db5874acp-1"),
    ("0x1.64006b7d6dc48p-2", "0x1.8692ee9c37120p+4", "0x1.bb19672334f57p-38"),
    ("0x1.06f4f268d05cep-1", "0x1.389b800000000p+17", "0x0.0p+0"),
    ("0x1.fffffffffffffp-1", "0x1.0000000000000p-1", "0x1.0000000000000p+0"),
    ("0x1.0000000000000p-53", "0x1.0000000000000p-1", "0x1.6a09e667f3bcdp-27"),
    ("0x1.8000000000000p+0", "0x1.0000000000000p-1", "0x1.3988e1409212ep+0"),
    ("0x1.01297d23ab683p-995", "0x1.0000000000000p-1", "0x1.6adc04a97c42ep-498"),
]
_PINNED = [tuple(float.fromhex(v) for v in row) for row in PINNED]
HOST_IS_GLIBC_FMA = (os.confstr("CS_GNU_LIBC_VERSION") or "").startswith("glibc") and all(
    _b(x ** y) == _b(want) for x, y, want in _PINNED)


def _canon64(u) -> np.ndarray:
    u = np.asarray(u, U64)
    return np.where((u & U64(M64 >> 1)) > U64(H.F64_INF), U64(H.F64_QNAN), u)


def _canon32(u) -> np.ndarray:
    u = np.asarray(u, U32)
    return np.where((u & U32(0x7FFFFFFF)) > U32(0x7F800000), U32(0x7FC00000), u)


def _f32(v: float) -> tuple[int, bool]:
    """``(cvtsd2ss(v), refused)``: the f32 word (NaN canonical) and whether ``Scalar::to<float>()`` refuses v."""
    with np.errstate(over="ignore"):
        w = int(_canon32(np.float64(v).astype(np.float32).view(U32)))
    return w, math.isfinite(v) and abs(v) > FLT_MAX


def _words(rng, n: int) -> np.ndarray:
    """Normal values over 2^-60 .. 2^60, their neighbours, raw words and special words."""
    normal = (rng.standard_normal(n) * np.exp2(rng.integers(-60, 60, n))).view(U64)
    near = normal ^ rng.integers(0, 1 << 8, n, dtype=U64)
    raw = rng.integers(0, 1 << 63, n, dtype=U64) * U64(2) + rng.integers(0, 2, n, dtype=U64)
    spec = np.array([0, 1 << 63, 1, 0x000FFFFFFFFFFFFF, 0x0010000000000000, H.F64_ONE, H.F64_NEG_ONE, 0x7FEFFFFFFFFFFFFF,
                     H.F64_INF, 0xFFF0000000000000, H.F64_QNAN, 0x7FF0000000000001, H.F64_FLT_MAX, H.F64_FLT_MAX + 1], U64)
    pick = rng.integers(0, 10, n)
    out = np.where(pick < 5, normal, np.where(pick < 7, near, np.where(pick < 9, raw, spec[rng.integers(0, len(spec), n)])))
    return out.astype(U64)


def _op(name, build, *params, ret=F64):
    return definition(name, 1, build, tuple((f"x{i}", t) for i, t in enumerate(params)), ret, register=False)


ADD = _op("TestF64Add", lambda C, a, b: [H.f64_add(C, a, b)], F64, F64)
SUB = _op("TestF64Sub", lambda C, a, b: [H.f64_sub(C, a, b)], F64, F64)
MUL = _op("TestF64Mul", lambda C, a, b: [H.f64_mul(C, a, b)], F64, F64)
DIV = _op("TestF64Div", lambda C, a, b: [H.f64_div(C, a, b)], F64, F64)
FMA = _op("TestF64Fma", lambda C, a, b, c: [H.f64_fma(C, a, b, c)], F64, F64, F64)
TO_F32 = _op("TestF64ToF32", lambda C, a: (lambda r: [r[0], [r[1]]])(H.f64_to_f32(C, a)), F64, ret=Tuple(F32, BL.BIT))
TO_F64 = _op("TestF32ToF64", lambda C, a: [H.f32_to_f64(C, a)], F32)
S12 = _op("TestS12ToF64", lambda C, k: [H.s12_to_f64(C, k)], Array(12, BL.BIT))
BIAS = bind(TO.TorchAdamBiasCorrection)
HALF = bind(TO.TorchPowHalf)
OUT = bind(TO.TorchAdamScalarsOut)
SCALARS = bind(TO.AdamWScalars)
SGD = bind(TO.TorchSgdScalar)


def _check(name: str, got, want, *ins) -> None:
    got, want = np.asarray(got).reshape(-1), np.asarray(want).reshape(-1)
    bad = np.flatnonzero(got != want)
    assert not bad.size, (f"{name}: {bad.size} of {got.size} differ, first at {bad[0]}: inputs "
                          f"{[hex(int(np.asarray(x).reshape(-1)[bad[0]])) for x in ins]}: {int(got[bad[0]]):#x} != {int(want[bad[0]]):#x}")


def test_f64_add_sub_mul_div_are_ieee() -> None:
    rng = np.random.default_rng(64)
    a, b = _words(rng, 600), _words(rng, 600)
    fa, fb = a.view(np.float64), b.view(np.float64)
    with np.errstate(all="ignore"):
        for D_, want in ((ADD, fa + fb), (SUB, fa - fb), (MUL, fa * fb), (DIV, fa / fb)):
            _check(D_.id, BS.evaluate_words(D_, a, b), _canon64(want.view(U64)), a, b)


def _fma_ref(a: int, b: int, c: int) -> int:
    fa, fb, fc = _d(a), _d(b), _d(c)
    if math.isnan(fa) or math.isnan(fb) or math.isnan(fc):
        return H.F64_QNAN
    sp = (math.copysign(1, fa) * math.copysign(1, fb)) < 0
    if math.isinf(fa) or math.isinf(fb):
        if fa == 0 or fb == 0 or (math.isinf(fc) and (fc < 0) != sp):
            return H.F64_QNAN
        return _b(-math.inf if sp else math.inf)
    if math.isinf(fc):
        return c
    try:
        return _b(_fma(fa, fb, fc))
    except OverflowError:
        return _b(math.inf if Fraction(fa) * Fraction(fb) + Fraction(fc) > 0 else -math.inf)


def test_f64_fma_and_casts() -> None:
    """The fma against exact arithmetic (half the addends near ``-a b``, so the sum cancels), the casts both ways, and the
    12-bit integer conversion on every input."""
    rng = np.random.default_rng(65)
    n = 600
    a, b, c = _words(rng, n), _words(rng, n), _words(rng, n)
    with np.errstate(all="ignore"):
        near = (-(a.view(np.float64) * b.view(np.float64))).view(U64) ^ rng.integers(0, 1 << 4, n, dtype=U64)
    c = np.where(rng.random(n) < 0.5, c, near).astype(U64)
    _check("fma", BS.evaluate_words(FMA, a, b, c), [_fma_ref(int(x), int(y), int(z)) for x, y, z in zip(a, b, c)], a, b, c)
    got = BS.evaluate_words(TO_F32, a).reshape(n, 2)
    want = [_f32(_d(x)) for x in a]
    _check("to_f32", got[:, 0], [w for w, _ in want], a)
    _check("to_f32 refused", got[:, 1], [r for _, r in want], a)
    with np.errstate(all="ignore"):
        w = np.where(rng.random(n) < 0.6, a.view(np.float64).astype(np.float32).view(U32), (a & U64(0xFFFFFFFF)).astype(U32))
    w = w.astype(U32)
    _check("to_f64", BS.evaluate_words(TO_F64, w), _canon64(w.view(np.float32).astype(np.float64).view(U64)), w)
    k = np.arange(4096, dtype=U64)
    _check("s12", BS.evaluate_words(S12, k), [_b(float(int(v) - 4096 * (int(v) >= 2048))) for v in k], k)


def test_twin_is_glibc_pow() -> None:
    """The twin's word at every pinned vector off its big range, and big exactly where glibc's word is below e^-512."""
    for x, y, want in _PINNED:
        p, big = pow_twin(x, y) if x != 1.0 else (1.0, False)
        assert big == (abs(y * math.log(x)) >= 512), (x.hex(), y.hex())
        assert big or _b(p) == _b(want), (x.hex(), y.hex(), p.hex(), want.hex())
        assert not big or want < math.exp(-512), (x.hex(), y.hex())


@pytest.mark.skipif(not HOST_IS_GLIBC_FMA, reason="the host's pow isn't glibc's __pow_fma at the pinned vectors")
def test_twin_is_host_pow() -> None:
    """On a host whose ``**`` is glibc's FMA pow: the twin, `one_minus_pow` and `pow_half` against CPython on random inputs."""
    rnd = random.Random(117)
    for _ in range(4000):
        x = rnd.choice([rnd.random(), 1 - 2.0 ** -rnd.uniform(1, 53), rnd.uniform(0, 4), 2.0 ** rnd.uniform(-1000, 1000)])
        y = rnd.choice([float(rnd.randint(1, 1 << 24)), float(rnd.randint(1, 20)), 0.5, rnd.uniform(-60, 60)])
        if x < TINY:
            continue
        p, big = pow_twin(x, y)
        assert big or _b(p) == _b(x ** y), (x.hex(), y.hex())
        if x < 1 and y >= 1:
            assert _b(one_minus_pow(x, y)) == _b(1 - x ** y), (x.hex(), y.hex())
        assert _b(pow_half(x)) == _b(x ** 0.5), x.hex()
    for beta in (0.0, -0.0, 5e-324, 2.0 ** -1000):
        for t in (1.0, 2.0, 1000.0, 2.0 ** 63, 2.0 ** 70):
            assert _b(one_minus_pow(beta, t)) == _b(1 - beta ** t), (beta, t)


def _betas(rng, n: int) -> np.ndarray:
    good = np.concatenate([np.full(n // 8, 0.9), np.full(n // 8, 0.95), rng.uniform(0, 1, n // 4),
                           1 - np.exp2(-rng.uniform(1, 53, n // 8)), np.exp2(-rng.uniform(1, 1074, n // 8))])
    odd = np.array([0.0, -0.0, 1.0, 1.5, -0.1, -2.0 ** -1074, math.nan, math.inf, -math.inf, 1 - 2.0 ** -53, 0.999])
    out = np.concatenate([good, odd[rng.integers(0, len(odd), n - len(good))]])
    rng.shuffle(out)
    return np.where(rng.random(n) < 0.05, _words(rng, n).view(np.float64), out).view(U64)


def _steps(rng, n: int) -> np.ndarray:
    ints = np.concatenate([rng.integers(0, 64, n // 4), rng.integers(0, 1 << 24, n // 4)]).astype(np.float32)
    odd = np.array([2.0 ** 24, 2.0 ** 24 - 1, 1e20, 2.0 ** 63, 2.0 ** 64, 3e38, 0.5, 1.25, -1.0, -0.5, -2.0, -0.0, math.nan,
                    math.inf, -math.inf, 1e-45], np.float32)
    out = np.concatenate([ints, odd[rng.integers(0, len(odd), n - len(ints))]])
    rng.shuffle(out)
    return out.view(U32)


def _beta_ok(b: float) -> bool:
    return 0.0 <= b < 1.0


def _whole(t: np.float32) -> bool:
    return bool(np.isfinite(t)) and t >= 1 and float(t) == math.floor(float(t))


def ref_bias(b1: float, b2: float, step_word: int) -> tuple[float, float, int, bool]:
    """`_multi_tensor_adam`'s ``RN(step + 1)``, its bias corrections and the checks AdamW makes."""
    with np.errstate(all="ignore"):
        t = np.array([step_word], U32).view(np.float32)[0] + np.float32(1)
    ok = _beta_ok(b1) and _beta_ok(b2) and _whole(t)
    y = float(t)
    return (one_minus_pow(b1, y) if ok else math.nan, one_minus_pow(b2, y) if ok else math.nan,
            int(_canon32(np.array([t]).view(U32))[0]), ok)


@pytest.mark.slow
def test_bias_corrections_match_host_code() -> None:
    """Both bias corrections against the twin on random betas (the recipe's, near 1, subnormal, invalid) and steps (integers,
    2^24 and past it, non-integers, negative, NaN): the two pow circuits themselves, which no cheaper test traces."""
    rng = np.random.default_rng(118)
    n = 1500
    b1, b2, st = _betas(rng, n), _betas(rng, n), _steps(rng, n)
    got = BS.evaluate_words(BIAS, b1, b2, st).reshape(n, 4)
    want = [ref_bias(_d(x), _d(y), int(s)) for x, y, s in zip(b1, b2, st)]
    ok = np.array([w[3] for w in want])
    _check("ok", got[:, 3], ok.astype(U64), b1, b2, st)
    _check("t", got[:, 2], [w[2] for w in want], st)
    assert ok.sum() > n // 3
    for j in range(2):
        _check(f"bc{j + 1}", got[ok, j], [_b(w[j]) for w, o in zip(want, ok) if o], b1[ok], b2[ok], st[ok])


@pytest.mark.slow
def test_pow_half_matches_host_code() -> None:
    """``x ** 0.5`` against the twin on the bias corrections a run reaches (down to 2^-53), across the positive normals, and at
    1, zero, subnormal, negative and non-finite x (ok 0): the pow circuit at y = 0.5."""
    rng = np.random.default_rng(119)
    n = 1200
    bc = [1 - b ** float(t) for b, t in zip(rng.uniform(0, 1, n // 3), rng.integers(1, 1 << 20, n // 3))]
    xs = np.concatenate([bc, 1 - np.exp2(-rng.uniform(1, 53, n // 6)), np.exp2(rng.uniform(-1022, 1023, n // 3)),
                         [1.0, 1 - 2.0 ** -53, 2.0 ** -53, 0.0, -0.0, 5e-324, TINY, -0.5, math.nan, math.inf, -math.inf]])
    xs = np.concatenate([xs, _words(rng, n - len(xs)).view(np.float64)]).view(U64)
    got = BS.evaluate_words(HALF, xs).reshape(-1, 2)
    fx = xs.view(np.float64)
    ok = (fx >= TINY) & np.isfinite(fx)
    _check("ok", got[:, 1], ok.astype(U64), xs)
    _check("x ** 0.5", got[ok, 0], [_b(pow_half(float(x))) for x in fx[ok]], xs[ok])


def ref_out(lr, b1, b2, eps, wd, bc1, s2, ok1, ok2) -> tuple[list[int], bool]:
    """The seven words `_multi_tensor_adam` hands its kernels from these Python floats, and the checks (at ``bc1 = 0``, where
    CPython raises, the IEEE quotient)."""
    with np.errstate(all="ignore"):
        q = float(np.float64(lr) / np.float64(bc1))
    vals = [1 - lr * wd, 1 - b1, b2, 1 - b2, s2, eps, q * -1]
    words, refused = zip(*(_f32(v) for v in vals))
    words, refused = list(words), list(refused)
    if wd == 0:
        words[0], refused[0] = TO.ONE, False
    ok = bool(ok1 and ok2 and 0.0 <= lr and 0.0 <= eps and 0.0 <= wd and bc1 != 0 and not any(refused))
    return words, ok


def _hyper(rng, n: int) -> list[np.ndarray]:
    """(lr, beta1, beta2, eps, wd) rows: training's recipe, random recipes, and wild words."""
    rows = []
    for _ in range(n):
        if rng.random() < 0.3:
            rows.append((3e-4, 0.9, 0.95, 1e-8, 0.1))
        else:
            rows.append((10 ** rng.uniform(-6, 0), rng.uniform(0, 1), rng.choice([0.0, rng.uniform(0, 1), 0.999]),
                         rng.choice([0.0, 1e-8, 10 ** rng.uniform(-12, -3)]), rng.choice([0.0, -0.0, 0.1, rng.uniform(0, 1)])))
    h = np.array(rows, np.float64).view(U64)
    wild = rng.random(h.shape) < 0.08
    h = np.where(wild, _words(rng, h.size).reshape(h.shape), h).astype(U64)
    return [np.ascontiguousarray(h[:, j]) for j in range(5)]


def test_scalars_out_match_host_code() -> None:
    rng = np.random.default_rng(120)
    n = 1500
    lr, b1, b2, eps, wd = _hyper(rng, n)
    bc1 = np.where(rng.random(n) < 0.8, (1 - rng.uniform(0, 1, n) ** rng.integers(1, 100, n)), _words(rng, n).view(np.float64))
    bc1 = np.where(rng.random(n) < 0.02, 0.0, bc1).view(U64)
    s2 = np.where(rng.random(n) < 0.8, rng.uniform(2.0 ** -27, 1, n), _words(rng, n).view(np.float64)).view(U64)
    ok1, ok2 = (rng.random(n) < 0.9).astype(U64), (rng.random(n) < 0.9).astype(U64)
    got = BS.evaluate_words(OUT, lr, b1, b2, eps, wd, bc1, s2, ok1, ok2).reshape(n, 8)
    want = [ref_out(*(_d(v[i]) for v in (lr, b1, b2, eps, wd, bc1, s2)), ok1[i], ok2[i]) for i in range(n)]
    _check("ok", got[:, 7], [w[1] for w in want], lr, b1, b2, eps, wd, bc1)
    assert sum(w[1] for w in want) > n // 3
    for j, name in enumerate(TO.SCALARS):
        _check(name, got[:, j], [w[0][j] for w in want], lr, b1, b2, eps, wd, bc1, s2)


def test_sgd_scalar() -> None:
    rng = np.random.default_rng(121)
    lr = np.concatenate([np.array([1e-2, 0.0, -0.0, -1e-2, math.nan, math.inf, 3.5e38, -3.5e38]).view(U64), _words(rng, 400)])
    got = BS.evaluate_words(SGD, lr).reshape(-1, 2)
    want = [_f32(-_d(v)) for v in lr]
    _check("nlr", got[:, 0], [w for w, _ in want], lr)
    _check("ok", got[:, 1], [int(not _d(v) < 0.0 and not r) for v, (_, r) in zip(lr, want)], lr)


def ref_scalars(lr, b1, b2, eps, wd, step_word) -> tuple[list[int], int, bool]:
    bc1, bc2, t, ok = ref_bias(b1, b2, step_word)
    s2 = pow_half(bc2) if ok else math.nan
    words, ok_out = ref_out(lr, b1, b2, eps, wd, bc1 if ok else 1.0, s2 if ok else 1.0, ok, ok and bc2 >= TINY)
    return words, t, ok_out


@pytest.mark.slow
def test_adamw_scalars_are_its_parts() -> None:
    """The composite on training's recipe and random recipes at the steps a run reaches: its wiring of the three Definitions
    (which outputs feed which inputs), checked against `ref_scalars`. It compiles the three pow circuits once more unless the
    two tests above ran in its process."""
    rng = np.random.default_rng(122)
    n = 300
    lr, b1, b2, eps, wd = _hyper(rng, n)
    st = _steps(rng, n)
    got = BS.evaluate_words(SCALARS, lr, b1, b2, eps, wd, st).reshape(n, 9)
    want = [ref_scalars(*(_d(v[i]) for v in (lr, b1, b2, eps, wd)), int(st[i])) for i in range(n)]
    ok = np.array([w[2] for w in want])
    _check("ok", got[:, 8], ok.astype(U64), lr, b1, b2, eps, wd, st)
    _check("t", got[:, 7], [w[1] for w in want], st)
    assert ok.sum() > n // 4
    for j, name in enumerate(TO.SCALARS):
        _check(name, got[ok, j], [w[0][j] for w, o in zip(want, ok) if o], lr[ok], b1[ok], b2[ok], eps[ok], wd[ok], st[ok])


def _capture(monkeypatch, torch) -> list[tuple[str, object]]:
    """Record every foreach call's Python scalars (the arguments torch converts with ``Scalar::to<float>()``)."""
    seen: list[tuple[str, object]] = []
    for name in ("_foreach_add_", "_foreach_mul_", "_foreach_lerp_", "_foreach_addcmul_", "_foreach_div_", "_foreach_addcdiv_"):
        fn = getattr(torch, name)

        def wrap(*args, _fn=fn, _name=name, **kw):
            for v in list(args[1:]) + list(kw.values()):
                if isinstance(v, (int, float)) or (isinstance(v, list) and v and isinstance(v[0], float)):
                    seen.append((_name, v))
            return _fn(*args, **kw)
        monkeypatch.setattr(torch, name, wrap)
    return seen


@pytest.mark.slow
@pytest.mark.skipif(not HOST_IS_GLIBC_FMA, reason="torch's bias corrections are the host's pow, not glibc's __pow_fma here")
def test_scalars_are_torch_adamw_and_sgd(monkeypatch) -> None:
    """torch 2.14's own AdamW(foreach=True) and SGD(foreach=True) steps on CPU at steps 1 .. 2^24 (the state's step set
    before each step): the scalars their foreach calls receive, each as ``Scalar::to<float>()`` makes it, and the step after,
    against the composite and `TorchSgdScalar_v1`: the test that binds the Definitions to torch's code rather than to this
    file's reading of it."""
    torch = pytest.importorskip("torch")
    recipes = [(3e-4, 0.9, 0.95, 1e-8, 0.1), (1e-3, 0.8, 0.999, 1e-6, 0.0)]
    starts = [0.0, 1.0, 2.0, 9.0, 999.0, 4095.0, 65535.0, 2.0 ** 24 - 1, 2.0 ** 24]
    rows, want = [], []
    for lr, b1, b2, eps, wd in recipes:
        p = torch.nn.Parameter(torch.tensor([0.5, -0.25, 1e-3, 2.0]))
        opt = torch.optim.AdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd, foreach=True)
        p.grad = torch.tensor([1e-3, -2e-3, 5e-4, 0.0])
        opt.step()
        for s in starts:
            opt.state[p]["step"] = torch.tensor(s)
            seen = _capture(monkeypatch, torch)
            opt.step()
            monkeypatch.undo()
            by = {}
            for name, v in seen:
                by.setdefault(name, []).append(v[0] if isinstance(v, list) else v)
            a = by["_foreach_mul_"][0] if wd != 0 else 1.0
            b2_ = by["_foreach_mul_"][-1]
            vals = [a, by["_foreach_lerp_"][0], b2_, by["_foreach_addcmul_"][0], by["_foreach_div_"][0],
                    by["_foreach_add_"][-1], by["_foreach_addcdiv_"][0]]
            want.append(([_f32(float(v))[0] for v in vals], _b(float(opt.state[p]["step"].item()))))
            rows.append((lr, b1, b2, eps, wd, np.float32(s).view(U32)))
    h = [np.array([r[j] for r in rows], np.float64).view(U64) for j in range(5)]
    st = np.array([r[5] for r in rows], U32)
    got = BS.evaluate_words(SCALARS, *h, st).reshape(len(rows), 9)
    assert got[:, 8].all()
    for i, (k, t) in enumerate(want):
        assert [int(v) for v in got[i, :7]] == k, (rows[i], [hex(int(v)) for v in got[i, :7]], [hex(v) for v in k])
        assert _b(float(np.array([got[i, 7]], U32).view(np.float32)[0])) == t
    lrs = np.array([1e-2, 3e-4], np.float64).view(U64)
    got = BS.evaluate_words(SGD, lrs).reshape(-1, 2)
    for i, lr in enumerate((1e-2, 3e-4)):
        p = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
        p.grad = torch.tensor([1e-3, -2e-3])
        opt = torch.optim.SGD([p], lr=lr, foreach=True)
        seen = _capture(monkeypatch, torch)
        opt.step()
        monkeypatch.undo()
        assert seen == [("_foreach_add_", -lr)] and int(got[i, 0]) == _f32(-lr)[0] and got[i, 1] == 1
