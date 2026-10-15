"""DeepSeek-V4-Pro's query norm, RoPE and SWA insert, and the indexer's query RoPE and FP8 quantisation (`dsv4_qnorm_insert`),
against models of their kernels.

* each word Definition word for word against a model written here from the kernels (vLLM d9105ea80's
  fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu and the CuTe DSL `IndexerQFp8Kernel`, read with their sm_90a SASS): every
  FADD, FMUL and FFMA rounded once from the exact rational, the warp's xor butterfly simulated lane by lane, e4m3 rounding from
  the exact value, the scale byte's truncation from the exact value; MUFU's `rsqrt.approx` and the libm pieces `NvLog2f_v1` /
  `NvExp2f_v1` are core's and FlashMLA's (`RsqrtApprox_v1`; `test_flashmla` pins the other two);
* the twin (`kernels.dsv4_qnorm_insert_twin`) word for word against the Definitions (`self_check`);
* the row the insert stores, read back from the paged cache where FlashMLA's decode reads it, through `FlashMlaKvRow_v1{448, 64,
  64}`: each element the exact product its bytes encode, within an e4m3 step of the token's own kv (or saturated at 448 scales);
* the indexer's outputs as `DsaMqaLogit_v1{H, D}`'s q and w: their types, and the logit within FP8 tolerance of
  `SCALE * sum_h relu(q_h . k) w_h ks` in float64;
* the edges: the scale byte's conversion on its edge words, a block with an infinite, a NaN or only zero elements, the
  indexer's infinite and NaN elements, the padded heads; bad statics refused by name;
* each Boolean `_v2` against its word view.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate, evaluate_batch, self_check
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.circuits.ops.mufu import RsqrtApprox
from verity_vllm.program import boolean as SW
from verity_vllm.program.kernels import dsv4_qnorm_insert_twin as T
from verity_vllm.program.registry import boolean_dsv4_qnorm_insert as BQ
from verity_vllm.program.registry import dsa as D
from verity_vllm.program.registry import dsv4_qnorm_insert as Q
from verity_vllm.program.registry import flashmla as F
from tests.program.test_dsa import _bf16 as _bf16_words
from tests.program.test_dsa import _butterfly, _ceil_ftz, _e4m3, _fadd, _fmax, _is_nan, _val
from tests.program.test_fp8_moe import _bf16 as _rn_bf16
from tests.program.test_fp8_moe import _ffma, _fmul, _q, _rn

SIGN, INF, M32 = 1 << 31, 0x7F800000, 0xFFFFFFFF
NAN = 0x7FC00000
S, X = Q.DSV4_INSERT, Q.DSV4_INDEXER
DEFS = {d.id: d for d in Q.DSV4_QNORM_INSERT_COMPOSITES}
TWIN_TARGETS = [t for family in T.KERNELS for t in T._targets(DEFS, family)]


# --- the model -------------------------------------------------------------------------------------------------------------------

def _fmin(a: int, b: int) -> int:
    """min.f32: a NaN operand yields the other, min(+0, -0) = -0."""
    if _is_nan(a):
        return 0x7FFFFFFF if _is_nan(b) else b
    if _is_nan(b):
        return a
    if _val(a) == _val(b):
        return a | b
    return a if _val(a) < _val(b) else b


def _byte(w: int) -> int:
    """The low byte of cvt.rzi.u32.f32 from the exact value: NaN and every negative to 0, past 2^32 - 1 to 0xFF."""
    if _is_nan(w) or w & SIGN:
        return 0
    if w == INF:
        return 0xFF
    return min(math.floor(_q(w)), M32) & 0xFF


def _rope(x: list[int], cs: list[int], DR: int, fused: bool = True) -> list[int]:
    """Pair p of the last DR elements by c = cs[p], s = cs[DR/2 + p]: FFMA(x_e, c, -FMUL(x_o, s)), FFMA(x_e, s, FMUL(x_o, c));
    with ``fused`` false, the source's products rounded and added (`x_e * c - x_o * s` uncontracted)."""
    out, base, half = list(x), len(x) - DR, DR // 2
    for p in range(half):
        xe, xo, c, s = x[base + 2 * p], x[base + 2 * p + 1], cs[p], cs[half + p]
        if fused:
            out[base + 2 * p] = _ffma(xe, c, _fmul(xo, s) ^ SIGN)
            out[base + 2 * p + 1] = _ffma(xe, s, _fmul(xo, c))
        else:
            out[base + 2 * p] = _fadd(_fmul(xe, c), _fmul(xo, s) ^ SIGN)
            out[base + 2 * p + 1] = _fadd(_fmul(xe, s), _fmul(xo, c))
    return out


def q_head(q: list[int], cs: list[int], eps: float, DR: int, fused: bool = True, fused_sumsq: bool = True) -> list[int]:
    """`processDeepseekV4Slot`'s query head (:205-297): lane l's sum of squares over elements 16l .. (FMUL, then FFMAs), `warpSum`
    (xor 16 .. 1), `rsqrtf(sum / D + eps)` as FFMA(sum, 1/D, eps) and MUFU, x * r, the RoPE, bf16.  ``fused_sumsq=False``
    is the deliberately wrong FMUL-then-FADD chain used by the sum-of-squares mutation test."""
    x = [v << 16 for v in q]
    L = len(x) // 32
    lanes = []
    for t in range(32):
        acc = _fmul(x[t * L], x[t * L])
        for i in range(t * L + 1, (t + 1) * L):
            acc = _ffma(x[i], x[i], acc) if fused_sumsq else _fadd(acc, _fmul(x[i], x[i]))
        lanes.append(acc)
    total = _butterfly(lanes, _fadd, (16, 8, 4, 2, 1))
    r = evaluate(RsqrtApprox, _ffma(total, D.f32_bits(1.0 / len(x)), D.f32_bits(eps)))[0]
    return [_rn_bf16(v) for v in _rope([_fmul(v, r) for v in x], cs, DR, fused)]


def kv_row(kv: list[int], cs: list[int], DR: int, QB: int, fused: bool = True) -> tuple[list[int], list[int], list[int]]:
    """`processDeepseekV4Slot`'s kv row (:300-371) on finite elements: the RoPE, per block amax (fmaxf over fabsf from +0),
    e = ceilf(log2f(max(amax, 1e-4) / 448)), the bytes of clamp(x * exp2f(-e)), the scale byte of e + 127, a 0 byte."""
    x = _rope([v << 16 for v in kv], cs, DR, fused)
    DN = len(x) - DR
    nope, scale = [], []
    for j in range(DN // QB):
        blk = x[j * QB:(j + 1) * QB]
        amax = 0
        for v in blk:
            amax = _fmax(amax, v & 0x7FFFFFFF)
        s = _rn(_q(_fmax(amax, D.F_1EM4)) / 448, 0)
        e = _ceil_ftz(evaluate(F.NvLog2f, s)[0])
        inv = evaluate(F.NvExp2f, e ^ SIGN)[0]
        nope += [_e4m3(_fmin(_fmax(_fmul(v, inv), Q.F_NEG448), Q.F_448)) for v in blk]
        scale.append(_byte(_fmax(_fmin(_fadd(e, Q.F_127), Q.F_255), 0)))
    return nope, scale + [0], [_rn_bf16(v) for v in x[DN:]]


def indexer_head(q: list[int], cs: list[int], w: int, DR: int, SCALE: float) -> tuple[list[int], int]:
    """`IndexerQFp8Kernel` for one (token, head) on finite elements: the RoPE through bf16, amax, the power-of-two scale's word
    `(bits(max(amax, 1e-4) * f32(1/448)) + 0x7FFFFF) & 0x7F800000`, its inverse `0x7F000000 - sb`, the bytes, the weight."""
    x = _rope([v << 16 for v in q], cs, DR)
    n = len(x)
    x = x[:n - DR] + [_rn_bf16(v) << 16 for v in x[n - DR:]]
    amax = 0
    for v in x:
        amax = _fmax(amax, v & 0x7FFFFFFF)
    sb = (_fmul(_fmax(amax, D.F_1EM4), Q.F_INV448) + 0x7FFFFF) & 0x7F800000
    inv = (0x7F000000 - sb) & M32
    return [_e4m3(_fmul(v, inv)) for v in x], _fmul(_fmul(w << 16, D.f32_bits(SCALE)), sb)


def _cs(rng, DR: int) -> list[int]:
    """A row of the f32 `cos_sin_cache` at a random position: DR/2 cosines, then DR/2 sines."""
    half = DR // 2
    theta = float(rng.integers(0, 1 << 16)) / 10000.0 ** (np.arange(half) / half)
    v = np.concatenate([np.cos(theta), np.sin(theta)]).astype(np.float32)
    return v.view(np.uint32).astype(int).tolist()


def _flat(out) -> list[int]:
    return [int(v) for v in out]


def _e4m3_value(b: int) -> Fraction:
    s, e, m = b >> 7, (b >> 3) & 15, b & 7
    assert not (e == 15 and m == 7), hex(b)
    v = Fraction(m, 8) * Fraction(2) ** -6 if e == 0 else (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7)
    return -v if s else v


QROPE = bind(Q.Dsv4QNormRope, D=S["DN"] + S["DR"], DR=S["DR"], EPS=S["EPS"])
KVROW = bind(Q.Dsv4KvRopeQuant, DN=S["DN"], DR=S["DR"], QB=S["QB"])
IHEAD = bind(Q.Dsv4IndexerQHead, D=X["D"], DR=X["DR"], SCALE=X["SCALE"])


# --- the constants ---------------------------------------------------------------------------------------------------------------

def test_the_constants_are_the_kernels():
    assert Q.F_INV448 == D.f32_bits(1.0 / 448.0)
    assert (Q.F_448, Q.F_NEG448, Q.F_127, Q.F_255) == tuple(D.f32_bits(v) for v in (448.0, -448.0, 127.0, 255.0))
    assert D.F_1EM4 == 0x38D1B717 == D.f32_bits(1e-4)


# --- the scale byte --------------------------------------------------------------------------------------------------------------

BYTE_EDGES = [0, SIGN, 1, 0x007FFFFF, 0x3F7FFFFF, 0x3F800000, 0x3FC00000, 0x437F0000, 0x437FFFFF, 0x43800000, 0x43808000,
              0x4B7FFFFF, 0x4B800000, 0x4F7FFFFF, 0x4F800000, 0x7F7FFFFF, INF, SIGN | INF, NAN, 0xFFFFFFFF, 0x7F800001, 0xBF800000,
              0xC3000000, Q.F_127, Q.F_255]


def _byte_words(rng) -> list[int]:
    """The edges, every (sign, exponent) at four mantissas, and random words."""
    out = list(BYTE_EDGES)
    for se in range(512):
        out += [se << 23 | m for m in (0, 1, 0x400000, 0x7FFFFF)]
    return out + rng.integers(0, 1 << 32, 4000, dtype=np.uint64).astype(int).tolist()


def test_the_scale_byte_on_its_edges():
    assert [Q.F32ToU32RzByte.evaluate(w) for w in BYTE_EDGES[:16]] == \
        [0, 0, 0, 0, 0, 1, 1, 255, 255, 0, 1, 0xFF, 0, 0, 0xFF, 0xFF]
    assert [Q.F32ToU32RzByte.evaluate(w) for w in BYTE_EDGES[16:]] == [0xFF, 0, 0, 0, 0, 0, 0, 127, 255]


def test_the_scale_byte_against_the_exact_truncation():
    words = _byte_words(np.random.default_rng(0))
    got = [Q.F32ToU32RzByte.evaluate(w) for w in words]
    assert got == [_byte(w) for w in words]
    assert [int(v) for v in T.u32_rz_byte(np.asarray(words, dtype=np.uint32))] == got


# --- the insert ------------------------------------------------------------------------------------------------------------------

def _q_cases(rng, Dd: int, DR: int) -> list[tuple[list[int], list[int]]]:
    cases = [(_bf16_words(rng, Dd, s), _cs(rng, DR)) for s in (1e-3, 0.3, 1.0, 1.0, 40.0, 3e4)]
    cases.append(([0] * Dd, _cs(rng, DR)))                                                # sum 0: rsqrt(eps)
    cases.append(([0x3F80] * (Dd // 2) + [0xBF80] * (Dd // 2), _cs(rng, DR)))
    return cases


def test_a_query_head_matches_the_kernel_model():
    rng = np.random.default_rng(1)
    for q, cs in _q_cases(rng, 512, 64):
        assert _flat(evaluate(QROPE, q, cs)) == q_head(q, cs, S["EPS"], 64)


@pytest.mark.parametrize("Dd,DR,eps", [(32, 8, 1e-6), (64, 16, 0.5), (128, 64, 1e-6)])
def test_a_query_head_at_other_statics_matches_the_model(Dd, DR, eps):
    rng = np.random.default_rng(Dd)
    fn = bind(Q.Dsv4QNormRope, D=Dd, DR=DR, EPS=eps)
    for q, cs in _q_cases(rng, Dd, DR)[:4]:
        assert _flat(evaluate(fn, q, cs)) == q_head(q, cs, eps, DR)


def test_the_query_sum_of_squares_is_fused():
    """At eps=2^-140, each lane's two widened bf16 inputs have subnormal, inexact squares.  FFMA and FMUL-then-FADD leave adjacent
    lane-sum words; after the butterfly and rsqrt this changes 32 bf16 outputs, so the Definition kills the unfused mutant."""
    eps = 2.0 ** -140
    q = [0x1D0D, 0x1D83] * 32
    cs = [D.f32_bits(1.0), 0]  # identity RoPE on the last pair
    fn = bind(Q.Dsv4QNormRope, D=64, DR=2, EPS=eps)
    got = _flat(evaluate(fn, q, cs))
    fused = q_head(q, cs, eps, 2)
    unfused = q_head(q, cs, eps, 2, fused_sumsq=False)
    assert got == fused
    assert [i for i, (a, b) in enumerate(zip(fused, unfused)) if a != b] == list(range(1, 64, 2))
    assert {(fused[i], unfused[i]) for i in range(1, 64, 2)} == {(0x3F98, 0x3F99)}


def _kv_cases(rng, Dd: int, DR: int) -> list[tuple[list[int], list[int]]]:
    cases = [(_bf16_words(rng, Dd, s), _cs(rng, DR)) for s in (1e-6, 1e-2, 1.0, 1.0, 50.0, 2e4)]
    mixed = _bf16_words(rng, Dd, 1.0)
    mixed[:64] = _bf16_words(rng, 64, 1e-3)                                               # blocks of other magnitudes
    mixed[64:128] = _bf16_words(rng, 64, 300.0)
    cases.append((mixed, _cs(rng, DR)))
    return cases


def test_a_kv_row_matches_the_kernel_model():
    rng = np.random.default_rng(2)
    for kv, cs in _kv_cases(rng, 512, 64):
        nope, scale, rope = kv_row(kv, cs, 64, 64)
        assert _flat(evaluate(KVROW, kv, cs)) == nope + scale + rope


@pytest.mark.parametrize("odd", [False, True], ids=["even", "odd"])
def test_the_rope_is_the_kernels_contraction(odd):
    """With x_e = x_o and s = c (s = -c), the even (odd) element is a difference of equal products: the FFMA leaves the first
    product's rounding error, which bf16 keeps, where the products rounded and added give +0.  Random rows cannot tell the two
    apart (they differ by an f32 ulp, under bf16's rounding)."""
    rng = np.random.default_rng(20 + odd)
    c = (0x3F000000 | rng.integers(1, 1 << 23, 32)).astype(int).tolist()
    cs = c + [v ^ SIGN if odd else v for v in c]
    q, kv = _bf16_words(rng, 512, 1.0), _bf16_words(rng, 512, 1.0)
    for x in (q, kv):
        x[449::2] = x[448::2]
    got = _flat(evaluate(QROPE, q, cs))
    assert got == q_head(q, cs, S["EPS"], 64) != q_head(q, cs, S["EPS"], 64, fused=False)
    nope, scale, rope = kv_row(kv, cs, 64, 64)
    assert _flat(evaluate(KVROW, kv, cs)) == nope + scale + rope
    assert rope != kv_row(kv, cs, 64, 64, fused=False)[2]
    k = 1 if odd else 0
    assert any(v & 0x7FFF for v in rope[k::2]) and any(v & 0x7FFF for v in got[448 + k::2])


def test_the_whole_insert_is_its_heads_the_pads_and_the_row():
    rng = np.random.default_rng(3)
    fn = bind(Q.Dsv4QNormRopeKvInsert, **{**S, "NH": 2, "NHP": 4})
    q, kv, cs = [_bf16_words(rng, 512, 2.0) for _ in range(2)], _bf16_words(rng, 512, 3.0), _cs(rng, 64)
    got = _flat(evaluate(fn, [v for h in q for v in h], kv, cs))
    nope, scale, rope = kv_row(kv, cs, 64, 64)
    assert got == q_head(q[0], cs, 1e-6, 64) + q_head(q[1], cs, 1e-6, 64) + [0] * (2 * 512) + nope + scale + rope


def test_a_block_with_an_infinite_element():
    """amax +inf: s, log2f and the ceiling stay +inf, the inverse exp2f(-inf) = +0, so the scale byte is fminf(inf, 255) = 255,
    the infinite element inf * 0 = NaN, clamped to -448 (0xFE), and every finite one a zero of its sign."""
    rng = np.random.default_rng(4)
    kv, cs = _bf16_words(rng, 512, 1.0), _cs(rng, 64)
    kv[3] = 0x7F80
    got = _flat(evaluate(KVROW, kv, cs))
    nope, scale, rope = kv_row([0 if i == 3 else v for i, v in enumerate(kv)], cs, 64, 64)
    assert got[:64] == [0xFE if i == 3 else 0x80 if kv[i] & 0x8000 else 0 for i in range(64)]
    assert got[448] == 255 and got[449:455] == scale[1:7] and got[455] == 0
    assert got[64:448] == nope[64:] and got[456:] == rope


def test_a_nan_element_is_minus_448():
    """fmaxf drops the NaN from amax, so the block's scale is the others'; the NaN element clamps to -448 (0xFE)."""
    rng = np.random.default_rng(5)
    kv, cs = _bf16_words(rng, 512, 1.0), _cs(rng, 64)
    nope, scale, rope = kv_row(kv[:70] + [0] + kv[71:], cs, 64, 64)
    kv[70] = 0x7FC1
    got = _flat(evaluate(KVROW, kv, cs))
    assert got[70] == 0xFE
    assert got[:70] + got[71:] == (nope + scale + rope)[:70] + (nope + scale + rope)[71:]


def test_a_zero_block_has_the_floor_scale():
    """amax +0 -> 1e-4: e = ceil(log2(1e-4 / 448)) = -22, so the byte is 105 and every element 0."""
    rng = np.random.default_rng(6)
    kv, cs = _bf16_words(rng, 512, 1.0), _cs(rng, 64)
    kv[:64] = [0] * 32 + [0x8000] * 32
    got = _flat(evaluate(KVROW, kv, cs))
    assert got[448] == 105 and got[:64] == [0] * 32 + [0x80] * 32
    assert Q.F32ToU32RzByte.evaluate(D.f32_bits(105.0)) == 105


# --- the cache row, read back as FlashMLA reads it --------------------------------------------------------------------------------

def test_the_stored_row_is_what_flashmla_dequantises():
    """The insert's row through the twin's `cache_rows` and `scatter` into a paged cache (block 4, 576 data bytes a row and 8
    scale bytes after the block's rows), read back at FlashMLA's offsets, through `FlashMlaKvRow_v1{448, 64, 64}`: the RoPE
    elements the insert's bf16, each NoPE element the exact product of its e4m3 byte and 2^(scale - 127), within an e4m3 step
    of the token's own kv (or saturated at 448 times the scale)."""
    rng = np.random.default_rng(7)
    DN, DR, QB, BS = S["DN"], S["DR"], S["QB"], 4
    row = bind(F.FlashMlaKvRow, DN=DN, DR=DR, QT=QB)
    kvs = [kv for kv, _ in _kv_cases(rng, DN + DR, DR)[:5]]
    css = [_cs(rng, DR) for _ in kvs]
    outs = [_flat(evaluate(KVROW, kv, cs)) for kv, cs in zip(kvs, css)]
    nope = np.asarray([o[:DN] for o in outs], dtype=np.uint8)
    scale = np.asarray([o[DN:DN + DN // QB + 1] for o in outs], dtype=np.uint8)
    rope = np.asarray([o[DN + DN // QB + 1:] for o in outs], dtype=np.uint16)
    tn, ts, tr = T.kv_rows(np.asarray(kvs, dtype=np.uint16), np.asarray(css, dtype=np.uint32), DR=DR, QB=QB)
    assert np.array_equal(tn, nope) and np.array_equal(ts, scale) and np.array_equal(tr, rope)
    data, sbytes = T.cache_rows(nope, scale, rope)
    slots, row_b, scale_b = [5, -1, 2, 11, 4], DN + 2 * DR, DN // QB + 1
    cache = T.scatter(np.full((3, BS * (row_b + scale_b)), 0xAA, dtype=np.uint8), slots, data, sbytes, BS)
    written = {divmod(s, BS) for s in slots if s >= 0}
    for b in range(3):
        for p in range(BS):
            if (b, p) not in written:
                assert (cache[b, p * row_b:(p + 1) * row_b] == 0xAA).all()
                assert (cache[b, BS * row_b + p * scale_b:BS * row_b + (p + 1) * scale_b] == 0xAA).all()
    for t, slot in enumerate(slots):
        if slot < 0:
            continue
        b, p = divmod(slot, BS)
        rd = cache[b, p * row_b:(p + 1) * row_b]
        rs = cache[b, BS * row_b + p * scale_b:][:DN // QB].tolist()
        assert cache[b, BS * row_b + p * scale_b + DN // QB] == 0
        rn, rr = rd[:DN].tolist(), rd[DN:].view("<u2").astype(int).tolist()
        assert rn == nope[t].tolist() and rr == rope[t].tolist() and rs == scale[t, :DN // QB].tolist()
        got = _flat(evaluate(row, rn, rs, rr, 1))
        assert got[DN:] == rr
        x = [v << 16 for v in kvs[t][:DN]]
        for d in range(DN):
            unit = Fraction(2) ** (rs[d // QB] - 127)
            v = _e4m3_value(rn[d]) * unit
            assert got[d] == (_rn_bf16(_rn(v, 0)) if v else (0x8000 if rn[d] & 0x80 else 0)), (t, d)
            xv = _q(x[d])
            if abs(xv) >= 448 * unit:
                assert v == (448 if xv > 0 else -448) * unit, (t, d)
            else:
                assert abs(v - xv) <= abs(xv) / 16 + unit / 1024, (t, d)


# --- the indexer's query ---------------------------------------------------------------------------------------------------------

def test_an_indexer_head_matches_the_kernel_model():
    rng = np.random.default_rng(8)
    cases = [(_bf16_words(rng, 128, s), _cs(rng, 64), _bf16_words(rng, 1)[0]) for s in (1e-3, 0.3, 1.0, 1.0, 40.0, 3e4)]
    cases.append(([0] * 128, _cs(rng, 64), 0x3F80))
    for q, cs, w in cases:
        qb, wt = indexer_head(q, cs, w, 64, X["SCALE"])
        assert _flat(evaluate(IHEAD, q, cs, w)) == qb + [wt]


def test_an_indexer_token_is_its_heads():
    rng = np.random.default_rng(9)
    H, Dd, DR = 8, 32, 16
    q, cs, w = [_bf16_words(rng, Dd) for _ in range(H)], _cs(rng, DR), _bf16_words(rng, H)
    got = _flat(evaluate(bind(Q.Dsv4IndexerQ, H=H, D=Dd, DR=DR, SCALE=X["SCALE"]), [v for r in q for v in r], cs, w))
    heads = [indexer_head(q[h], cs, w[h], DR, X["SCALE"]) for h in range(H)]
    assert got == [v for qb, _ in heads for v in qb] + [wt for _, wt in heads]


def test_an_indexer_head_with_an_infinite_element():
    """amax +inf: the scale's word is +inf and its inverse 0x7F000000 - 0x7F800000 = -inf; with no clamp each finite nonzero
    element is -sign * 448 (the saturating cvt), each zero NaN (0x7F), and the weight infinite."""
    rng = np.random.default_rng(10)
    q, cs = _bf16_words(rng, 128, 1.0), _cs(rng, 64)
    q[5], q[6] = 0x7F80, 0
    got = _flat(evaluate(IHEAD, q, cs, 0x3F80))
    assert got[5] == 0xFE and got[6] == 0x7F
    nope = [0x7F if v & 0x7FFF == 0 else 0x7E if v & 0x8000 else 0xFE for v in q[:64]]
    assert got[:64] == nope and got[128] == INF


def test_an_indexer_nan_element():
    rng = np.random.default_rng(11)
    q, cs, w = _bf16_words(rng, 128, 1.0), _cs(rng, 64), 0x3F80
    qb, wt = indexer_head(q[:9] + [0] + q[10:], cs, w, 64, X["SCALE"])
    q[9] = 0xFFC0
    got = _flat(evaluate(IHEAD, q, cs, w))
    assert got[9] == 0x7F and got[:9] + got[10:] == (qb + [wt])[:9] + (qb + [wt])[10:]


def test_the_indexer_outputs_are_the_logit_kernels_q_and_w():
    """`Dsv4IndexerQ_v1{64, 128}`'s outputs have `DsaMqaLogit_v1{64, 128}`'s q and w types (DSV4-IDX's `DsaMqaLogit_v3` has the
    same signature), and through the logit at 8 heads of 32 they give `SCALE * sum_h relu(q_h . k) w_h ks` within FP8's error:
    the weight carries the scale the bytes divide out."""
    out = bind(Q.Dsv4IndexerQ, **X).signature()[1]
    (_, qt), (_, wt), *_ = bind(D.DsaMqaLogit, H=X["H"], D=X["D"]).signature()[0]
    assert tuple(out.fields) == (qt, wt)
    rng = np.random.default_rng(12)
    H, Dd, DR = 8, 32, 16
    tok = bind(Q.Dsv4IndexerQ, H=H, D=Dd, DR=DR, SCALE=X["SCALE"])
    lg = bind(D.DsaMqaLogit, H=H, D=Dd)
    for _ in range(4):
        q, cs, w = [_bf16_words(rng, Dd) for _ in range(H)], _cs(rng, DR), _bf16_words(rng, H, 0.5)
        got = _flat(evaluate(tok, [v for r in q for v in r], cs, w))
        q8, w8 = got[:H * Dd], got[H * Dd:]
        k = rng.integers(0, 0x7F, Dd).astype(int) | (rng.integers(0, 2, Dd).astype(int) << 7)
        k = [int(v) for v in np.where((k & 0x7F) == 0x7F, 0x7E, k)]
        ks = D.f32_bits(0.01)
        logit = _val(evaluate(lg, q8, w8, k, ks)[0])
        rows = [_rope([u << 16 for u in r], cs, DR) for r in q]
        x = np.asarray([[_val(v) for v in r[:Dd - DR] + [_rn_bf16(u) << 16 for u in r[Dd - DR:]]] for r in rows])
        kv = np.asarray([float(_e4m3_value(b)) for b in k])
        wv = np.asarray([_val(v << 16) for v in w])
        dots = x @ kv
        want = float(np.float32(X["SCALE"])) * 0.01 * float(np.sum(np.maximum(dots, 0) * wv))
        bound = float(np.float32(X["SCALE"])) * 0.01 * float(np.sum(np.abs(x) @ np.abs(kv) * np.abs(wv))) / 8
        assert abs(logit - want) <= bound, (logit, want, bound)


# --- the twin --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", TWIN_TARGETS, ids=[t.id for t in TWIN_TARGETS])
def test_the_twin_is_the_reference(target):
    sc = self_check(target, kernel=T.KERNEL, n=24, seed=0)
    assert sc.ok, str(sc)
    assert sc.declined == 0, str(sc)


def test_the_twin_pads_heads_with_zero():
    rng = np.random.default_rng(13)
    q = np.asarray([[_bf16_words(rng, 64) for _ in range(2)]], dtype=np.uint16)
    kv, cs = np.asarray([_bf16_words(rng, 64)], dtype=np.uint16), np.asarray([_cs(rng, 16)], dtype=np.uint32)
    out = T.insert(q, kv, cs, NHP=5, DR=16, QB=16, EPS=1e-6)[0]
    assert out.shape == (1, 5, 64) and not out[:, 2:].any() and out[:, :2].any()


# --- bad statics -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fn,kw,needle", [
    (Q.Dsv4QNormRope, dict(D=48, DR=16, EPS=1e-6), "D=48 must be a power of two"),
    (Q.Dsv4QNormRope, dict(D=16, DR=8, EPS=1e-6), "D=16 must be a power of two"),
    (Q.Dsv4QNormRope, dict(D=64, DR=15, EPS=1e-6), "DR=15 must be even"),
    (Q.Dsv4QNormRope, dict(D=64, DR=128, EPS=1e-6), "at most D=64"),
    (Q.Dsv4KvRopeQuant, dict(DN=48, DR=16, QB=32), "QB=32 must divide DN=48"),
    (Q.Dsv4QNormRopeKvInsert, dict(NH=0, NHP=4, DN=48, DR=16, QB=16, EPS=1e-6), "NH=0 must be in [1, NHP=4]"),
    (Q.Dsv4QNormRopeKvInsert, dict(NH=5, NHP=4, DN=48, DR=16, QB=16, EPS=1e-6), "NH=5 must be in [1, NHP=4]"),
    (Q.Dsv4IndexerQHead, dict(D=32, DR=3, SCALE=0.5), "DR=3 must be even"),
])
def test_bad_statics_are_refused_by_name(fn, kw, needle):
    with pytest.raises(ValueError, match=needle.replace("[", r"\[").replace("]", r"\]")):
        bind(fn, **kw).body


# --- the Boolean Definitions -----------------------------------------------------------------------------------------------------

def test_the_boolean_scale_byte_is_its_word():
    target = bind(BQ.F32ToU32RzByteV2)
    word = word_of(target)
    assert word.id == Q.F32ToU32RzByte.id and SW.boolean_version(word).id == target.id
    x = np.asarray(_byte_words(np.random.default_rng(14)), dtype=np.uint64)
    got = np.asarray(evaluate_words(target, x), dtype=np.uint64).reshape(-1)
    assert got.tolist() == [_byte(int(w)) for w in x]


BOOLEAN = [
    (bind(BQ.Dsv4QNormRopeV2, D=32, DR=8, EPS=1e-6), T._sample_q),
    (bind(BQ.Dsv4KvRopeQuantV2, DN=16, DR=16, QB=16), T._sample_kv),
    (bind(BQ.Dsv4QNormRopeKvInsertV2, NH=1, NHP=2, DN=16, DR=16, QB=16, EPS=1e-6), T._sample_insert),
    (bind(BQ.Dsv4IndexerQHeadV2, D=32, DR=16, SCALE=0.75), T._sample_head),
    (bind(BQ.Dsv4IndexerQV2, H=2, D=32, DR=16, SCALE=0.75), T._sample_token),
]


@pytest.mark.slow
@pytest.mark.parametrize("target,sample", BOOLEAN, ids=[t.id for t, _ in BOOLEAN])
def test_the_boolean_definitions_are_their_words(target, sample):
    """Each traced circuit (120,000 to 380,000 ANDs, 10 to 50 s to trace and run) against its word Definition on the twin's
    sampler, edge words in about a third of the rows: what the word tests above cannot reach is the Boolean callees (the
    GPU-NaN f32 operations, `NvLog2f_v2`, the byte) composed as these bodies compose them."""
    word = word_of(target)
    assert SW.boolean_version(word).id == target.id
    args = sample(np.random.default_rng(15), 64, word.signature()[0], **dict(word.bindings))
    got = np.asarray(evaluate_words(target, *args), dtype=np.uint64).reshape(64, -1)
    want = np.asarray(evaluate_batch(word, *args, kernel="reference"), dtype=np.uint64).reshape(64, -1)
    assert np.array_equal(got, want)
