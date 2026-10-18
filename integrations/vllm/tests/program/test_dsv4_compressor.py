"""DeepSeek-V4-Pro's head-512 compressor on the H200 (`dsv4_compressor`, `boolean_dsv4_compressor`, `kernels.dsv4_compressor_twin`).

* the twin (the kernels written again over numpy words, each amax in its kernel's own lane order) word for word against the
  reference on every binding it registers;
* the UE8M0 steps on every byte, and the scale byte as the exponent rounded up;
* the words the hardware makes of special values: an all-zero or all-NaN group takes the 1e-4 floor's byte 114, a NaN quantises
  to -448 (0xFE), an infinite amax gives the byte 255 whose inverse is -inf;
* the ratio-4 window at position 3, whose skipped rows are not rows of score -inf;
* the row as the decode reads it (`flashmla.FlashMlaKvRow_v1` at the decode's statics), and both compressors within a float64
  restatement of the math;
* the state score both compressors read (`dsv4_indexer.Dsv4CompressorScore_v1` at W = 1024 and 512);
* the small Boolean Definitions against their word views; the rest are circuit-check's (`boolean_dsv4_compressor.circuit-check.py`).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from verity.ml.circuits.boolean.forms import Circuit, const_bits
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate, evaluate_batch, self_check
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.circuits.silicon import fp as FP
from verity_vllm.program import boolean as SW
from verity_vllm.program.kernels import dsv4_compressor_twin as T
from verity_vllm.program.kernels import flashmla_twin as FT
from verity_vllm.program.kernels.fp8_twins import np_f32_to_e4m3_sat
from verity_vllm.program.registry import boolean_dsv4_compressor as BC
from verity_vllm.program.registry import boolean_dsv4_indexer as BI
from verity_vllm.program.registry import dsv4_compressor as C
from verity_vllm.program.registry import dsv4_indexer as I
from verity_vllm.program.registry import flashmla as F

U = np.uint32
S = C.DSV4_COMPRESSOR
DEFS = {d.id: d for d in C.DSV4_COMPRESSOR_COMPOSITES}
TWIN_TARGETS = [t for family in T.KERNELS_FN for t in T._targets(DEFS, family)]
NAN, PINF, NINF, SIGN = 0x7FFFFFFF, 0x7F800000, 0xFF800000, 0x80000000
ONE, MONE = 0x3F800000, 0xBF800000
SPECIAL = [0, SIGN, PINF, NINF, 0x7FC00000, NAN, 0xFFFFFFFF, 1, 0x80000001, 0x007FFFFF, 0x00800000, ONE, MONE, 0x38D1B717,
           0x7F000000, 0x7F7FFFFF, 0xFF7FFFFF]


def _bits(v: float) -> int:
    return int(np.float32(v).view(U))


def _f64(w) -> np.ndarray:
    return np.asarray(w, dtype=U).view(np.float32).astype(np.float64)


def _bf16_f64(h) -> np.ndarray:
    return _f64(np.asarray(h, dtype=U) << U(16))


def _flat(out) -> list[int]:
    return [int(v) for v in out]


# --- the twin --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", TWIN_TARGETS, ids=[t.id for t in TWIN_TARGETS])
def test_the_twin_is_the_reference(target):
    sc = self_check(target, kernel=T.KERNEL, n=24, seed=0)
    assert sc.ok, str(sc)
    assert sc.declined == 0, str(sc)


# --- the UE8M0 steps ---------------------------------------------------------------------------------------------------------------

def test_the_scale_byte_is_the_exponent_rounded_up():
    """For every positive normal scale s, `Ue8m0CeilByte_v1` is ceil(log2 s) + 127 (powers of two exactly, every other word one
    up), 255 past 2^127 and for +inf; the floor 1e-4 gives 114."""
    rng = np.random.default_rng(0)
    s = np.concatenate([2.0 ** np.arange(-126, 128), np.nextafter(2.0 ** np.arange(-126, 127), np.inf, dtype=np.float32),
                        rng.uniform(0.5, 1.0, 4000) * 2.0 ** rng.integers(-125, 129, 4000)]).astype(np.float32)
    for v in s[np.isfinite(s)]:
        m, e = math.frexp(float(v))
        assert C.Ue8m0CeilByte.evaluate(_bits(v)) == min((e if m > 0.5 else e - 1) + 127, 255), float(v)
    assert [C.Ue8m0CeilByte.evaluate(w) for w in (0x38D1B717, ONE, 0x7F7FFFFF, PINF)] == [114, 127, 255, 255]


def test_the_inverse_scale_is_the_reciprocal_until_the_sign_bit():
    """`Ue8m0InvScale_v1(e)` is 2^(127 - e) for e <= 253, +0 at 254 (2^-127 is below the normal range: the word has no
    exponent left) and -inf at 255 (254 - 255 borrows into the sign bit)."""
    got = [C.Ue8m0InvScale.evaluate(e) for e in range(256)]
    assert got[:254] == [_bits(2.0 ** (127 - e)) for e in range(254)]
    assert got[254:] == [0, NINF]


# --- the special values ------------------------------------------------------------------------------------------------------------

QUANT8 = bind(C.Dsv4MlaQuantGroup, QB=8)


def test_the_group_quantisation_at_its_special_words():
    """An all-zero group takes the floor 1e-4's byte 114 and keeps each zero's sign; a NaN quantises to -448 (0xFE: `max.f32(NaN,
    -448)` is -448, and the min after it a setp / selp that never sees the NaN), and an all-NaN group has the floor's scale; an
    infinite word makes the scale +inf, byte 255, inverse -inf, so every finite word saturates to the opposite sign and a zero
    (0 * inf) is NaN, -448."""
    assert _flat(evaluate(QUANT8, [0, SIGN] * 4)) == [0x00, 0x80] * 4 + [114]
    assert _flat(evaluate(QUANT8, [0x7FC00000] * 8)) == [0xFE] * 8 + [114]
    rng = np.random.default_rng(1)
    x = _flat(np.asarray(rng.standard_normal(8), np.float32).view(U))
    with_nan = _flat(evaluate(QUANT8, [NAN] + x[1:]))
    assert with_nan[0] == 0xFE and with_nan[1:] == _flat(evaluate(QUANT8, [0] + x[1:]))[1:]
    assert _flat(evaluate(QUANT8, [PINF, ONE, MONE, 0, SIGN, NINF, 0x7F000000, 0xFF000000])) == \
        [0xFE, 0xFE, 0x7E, 0xFE, 0xFE, 0x7E, 0xFE, 0x7E, 255]


def test_an_amax_of_448_times_a_power_of_two_takes_that_scale():
    """amax = 448 * 2^k divides to 2^k exactly, so its byte is k + 127 and it stores 448 (0x7E); one bf16 step above, the scale
    doubles.  Below 2^-13 the floor 1e-4 holds the byte at 114."""
    rng = np.random.default_rng(2)
    rest = _flat((np.asarray(rng.uniform(-1, 1, 7), np.float32) * np.float32(2.0 ** -5)).view(U))
    assert _flat(evaluate(QUANT8, [_bits(448.0 * 2.0 ** -20)] + rest))[8] == 114
    for k in (-12, -3, 0, 9):
        top = _bits(448.0 * 2.0 ** k)
        got = _flat(evaluate(QUANT8, [top] + rest))
        assert got[0] == 0x7E and got[8] == k + 127
        assert _flat(evaluate(QUANT8, [top + 0x10000] + rest))[8] == k + 128


def test_the_skipped_rows_at_position_3_are_not_rows_of_score_minus_inf():
    """At position 3 the kernel skips the window's first 4 rows (cutedsl:233) rather than reading them as score -inf: a leading -inf
    row would make (m - m') = -inf - -inf NaN, and the column NaN."""
    rng = np.random.default_rng(3)
    kv, score = (_flat(np.asarray(rng.standard_normal(4) * s, np.float32).view(U)) for s in (1.0, 2.0))
    four = evaluate(bind(C.Dsv4MlaC4Column, ROWS=4), kv, score)[0]
    assert four & 0x7F800000 != 0x7F800000
    assert evaluate(bind(C.Dsv4MlaC4Column, ROWS=8), [0] * 4 + kv, [NINF] * 4 + score)[0] == NAN


# --- the row the decode reads ------------------------------------------------------------------------------------------------------

def _rows(rng, n: int, rows: int):
    kv = (rng.standard_normal((n, rows, S["D"])) * 2.0 ** rng.integers(-6, 6, (n, 1, 1))).astype(np.float32).view(U)
    score = (rng.standard_normal((n, rows, S["D"])) * 2.0).astype(np.float32).view(U)
    w = ((rng.standard_normal((n, S["D"])) * 0.2 + 1.0).astype(np.float32).view(U) >> U(16)).astype(np.uint16)
    return kv, score, w, T._cs(rng, n, S["QB"])


def test_the_decode_reads_the_row_back():
    """`FlashMlaKvRow_v1` at the decode's statics (DN 448, DR 64, QT 64) on a compressed row, its pad byte left off: each NoPE
    element the bf16-rounded normed word within half an e4m3 step at its group's scale, the RoPE elements as stored."""
    DN, DR, QT = F.FLASHMLA_DSV4["DN"], F.FLASHMLA_DSV4["DR"], F.FLASHMLA_DSV4["QT"]
    assert (DN, DR, QT) == (S["D"] - S["QB"], S["QB"], S["QB"])
    rng = np.random.default_rng(4)
    kv, score, w, cs = _rows(rng, 3, 8)
    c = T.c4_columns(np.swapaxes(kv, -1, -2), np.swapaxes(score, -1, -2))
    q = _bf16_f64(T.bf16(T.rms_norm(c, w, 4, S["QB"], S["EPS"])))[:, :DN]
    nope, rope, scale = T.compress_c4(kv, score, w, cs, S["QB"], S["EPS"])
    assert not scale[:, -1].any()
    row = bind(F.FlashMlaKvRow, DN=DN, DR=DR, QT=QT)
    for i in range(3):
        got = np.asarray(evaluate(row, _flat(nope[i]), _flat(scale[i, :-1]), _flat(rope[i]), 1), dtype=np.uint16)
        assert np.array_equal(got[DN:], rope[i])
        unit = np.repeat(2.0 ** (scale[i, :-1].astype(np.int64) - 127), QT)
        assert (np.abs(_bf16_f64(got[:DN]) - q[i]) <= np.maximum(np.abs(q[i]) * 2.0 ** -4, unit * 2.0 ** -10)).all()


def _steps(a: np.ndarray, b: np.ndarray) -> int:
    """The largest distance between e4m3 bytes in representable steps."""
    key = lambda u: np.where(u >= 0x80, 0xFF - u.astype(np.int64), u.astype(np.int64) + 0x80)  # noqa: E731
    return int(np.abs(key(a) - key(b)).max())


def _restated(kv, score, w, cs, QB: int, eps: float):
    """The row in float64: the softmax over the rows, the weighted sum, the RMSNorm, the rotation of the last QB, each element
    rounded to bf16 as the kernels store it, and the UE8M0 quantisation (2^ceil(log2(max(amax / 448, 1e-4))))."""
    k, s = _f64(kv), _f64(score)
    p = np.exp(s - s.max(axis=-2, keepdims=True))
    c = (p / p.sum(axis=-2, keepdims=True) * k).sum(axis=-2)
    n = c / np.sqrt((c * c).mean(axis=-1, keepdims=True) + eps) * _bf16_f64(w)
    D, h = n.shape[-1], QB // 2
    cos, sin = _f64(cs)[..., :h], _f64(cs)[..., h:]
    x0, x1 = n[..., D - QB::2].copy(), n[..., D - QB + 1::2].copy()
    rot = np.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], axis=-1).reshape(*n.shape[:-1], QB)
    size = np.repeat(np.abs(x0) + np.abs(x1), 2, axis=-1)
    groups = _bf16_f64(T.bf16(n[..., :D - QB].astype(np.float32).view(U))).reshape(*n.shape[:-1], -1, QB)
    e = np.ceil(np.log2(np.maximum(np.abs(groups).max(axis=-1) / 448.0, 1e-4))).astype(np.int64)
    q = np_f32_to_e4m3_sat((groups / 2.0 ** e[..., None]).astype(np.float32))
    return q.reshape(*n.shape[:-1], D - QB), rot, size, (e + 127).astype(np.uint8)


@pytest.mark.parametrize("ratio", [4, 128])
def test_the_row_within_a_float64_restatement(ratio):
    """Both compressors at the model's statics against `_restated`: each NoPE byte within 1 e4m3 step and the same scale bytes (the
    kernels' ex2.approx and rsqrt.approx move the normed words by a few f32 ulps), each RoPE element within 1 bf16 step of its
    float64 value, or within 2^-20 of its pair's size where the rotation cancels."""
    rng = np.random.default_rng(5 + ratio)
    rows = S["ROWS"] if ratio == 4 else S["W"] * S["R"]
    kv, score, w, cs = _rows(rng, 4, rows)
    if ratio == 4:
        nope, rope, scale = T.compress_c4(kv, score, w, cs, S["QB"], S["EPS"])
    else:
        nope, rope, scale = T.compress_c128(kv, score, w, cs, S["W"], S["R"], S["QB"], S["EPS"])
    q, rot, size, e = _restated(kv, score, w, cs, S["QB"], S["EPS"])
    assert _steps(nope, q) <= 1
    assert np.array_equal(scale[:, :-1], e)
    assert (np.abs(_bf16_f64(rope) - rot) <= np.maximum(np.abs(rot) * 2.0 ** -7, size * 2.0 ** -20)).all()


def test_the_twin_rows_are_the_definitions_at_the_models_statics():
    """One ratio-4 row at D = 512 through the reference, against the twin the float64 test reads (the self-check binds the
    compressors at D = 32)."""
    rng = np.random.default_rng(6)
    kv, score, w, cs = _rows(rng, 1, S["ROWS"])
    fn = bind(C.Dsv4MlaCompressC4, ROWS=S["ROWS"], D=S["D"], QB=S["QB"], EPS=S["EPS"])
    want = np.concatenate([np.asarray(p).reshape(-1) for p in T.compress_c4(kv, score, w, cs, S["QB"], S["EPS"])])
    assert _flat(evaluate(fn, kv.reshape(-1), score.reshape(-1), w.reshape(-1), cs.reshape(-1))) == _flat(want)


# --- the state score ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("W", [2 * S["D"], S["D"]])
def test_the_state_score_at_both_widths(W):
    """`save_partial_states`' score row for this compressor (coff * 512 wide: 1024 on ratio 4, 512 on ratio 128) is
    `dsv4_indexer.Dsv4CompressorScore_v1{W}`: add.f32 of the score and the APE row."""
    rng = np.random.default_rng(W)
    score = np.where(rng.random(W) < 0.1, rng.choice(np.array(SPECIAL, U), W), FT._f32(rng, (W,), scale=4.0, p=0.0)).astype(U)
    ape = FT._f32(rng, (W,), scale=0.1, p=0.0)
    assert _flat(evaluate(bind(I.Dsv4CompressorScore, W=W), score, ape)) == _flat(FT.fadd(score, ape))


# --- the statics -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fn,kw,needle", [
    (C.Dsv4MlaRmsNorm, dict(RATIO=8, D=512, QB=64, EPS=1e-6), "RATIO=8"),
    (C.Dsv4MlaRmsNorm, dict(RATIO=4, D=48, QB=8, EPS=1e-6), "D=48"),
    (C.Dsv4MlaNormRopeQuant, dict(RATIO=4, D=64, QB=64, EPS=1e-6), "QB=64"),
    (C.Dsv4MlaQuantGroup, dict(QB=12), "QB=12"),
    (C.Dsv4MlaC128Column, dict(W=3, R=4), "W=3"),
    (C.Dsv4MlaRope, dict(ROPE=3), "ROPE=3"),
    (C.Dsv4MlaC4Column, dict(ROWS=0), "ROWS=0"),
    (C.Dsv4MlaRmsNorm, dict(RATIO=128, D=512, QB=64, EPS=0.0), "EPS"),
], ids=["ratio", "width", "group", "quant-group", "warps", "rope", "rows", "eps"])
def test_bad_statics_are_refused_by_name(fn, kw, needle):
    with pytest.raises(ValueError, match=needle):
        bind(fn, **kw).body


# --- the Boolean Definitions -------------------------------------------------------------------------------------------------------

def _is_its_word(target, args, n: int) -> None:
    word = word_of(target)
    assert SW.boolean_version(word).id == target.id
    got = np.asarray(evaluate_words(target, *args), dtype=np.uint64).reshape(n, -1)
    want = np.asarray(evaluate_batch(word, *args, kernel="reference"), dtype=np.uint64).reshape(n, -1)
    assert np.array_equal(got, want)


def test_the_boolean_ue8m0_steps_are_their_words():
    rng = np.random.default_rng(7)
    s = np.concatenate([np.asarray(SPECIAL, np.uint64), rng.integers(0, 1 << 32, 1000, dtype=np.uint64)])
    _is_its_word(bind(BC.Ue8m0CeilByteV2), [s], len(s))
    _is_its_word(bind(BC.Ue8m0InvScaleV2), [np.arange(256, dtype=np.uint64)], 256)


@pytest.mark.parametrize("zero", [0, SIGN])
@pytest.mark.parametrize("side", [0, 1])
def test_an_fma_of_a_constant_zero_product_is_fp_f32_fma(zero, side):
    """The same forms as `fp.f32_fma`, every AND one it already made."""
    C_ = Circuit(cse=True)
    x, c = C_.input("x", 32), C_.input("c", 32)
    a, b = (const_bits(zero, 32), x) if side == 0 else (x, const_bits(zero, 32))
    want = [f.key() for f in FP.f32_fma(C_, a, b, c, nan_word=BI.PTX)]
    n = len(C_.kind)
    assert BC.zero_word(a) or BC.zero_word(b)
    assert [f.key() for f in BC.GateOps(C_).fma(a, b, c)] == want and len(C_.kind) == n


@pytest.mark.parametrize("target,family", [
    (bind(BC.Dsv4MlaQuantGroupV2, QB=8), "Dsv4MlaQuantGroup_v1"),
    (bind(BC.Dsv4MlaRopeV2, ROPE=4), "Dsv4MlaRope_v1"),
    (bind(BC.Dsv4MlaC4ColumnV2, ROWS=2), "Dsv4MlaC4Column_v1"),
], ids=["quant", "rope", "c4-column"])
def test_the_small_boolean_definitions_are_their_words(target, family):
    st = dict(target.bindings)
    _is_its_word(target, T.SAMPLES[family](np.random.default_rng(8), 64, None, **st), 64)


@pytest.mark.slow
def test_the_boolean_ratio_128_row_is_its_word():
    """The ratio-128 compressor on bits at D = 16 over 2 warps of 2 rows, about 30 s: its Calls of the column, the norm, the
    groups, the rotation and the pad byte's constant bits wired as the word body's are (circuit-check holds each leaf to its word
    on its own vectors, not this composition on the twin's state rows)."""
    st = dict(W=2, R=2, D=16, QB=8, EPS=1e-6)
    target = bind(BC.Dsv4MlaCompressC128V2, **st)
    _is_its_word(target, T.SAMPLES["Dsv4MlaCompressC128_v1"](np.random.default_rng(9), 8, None, **st), 8)
