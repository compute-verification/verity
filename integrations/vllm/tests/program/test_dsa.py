"""GLM-5.3's DeepSeek Sparse Attention indexer (`dsa`) against models of its kernels.

* each word Definition word for word against a model written here from the kernels (vLLM d9105ea80, Triton 3.7.1's sm_90 PTX,
  DeepGEMM 8b1392b9): every FADD, FMUL and FFMA rounded once from the exact rational, the warp butterflies simulated lane by
  lane (`shfl.sync.bfly`), `__nv_log2f` with both of its branches, e4m3 rounding from the exact value; MUFU's `rsqrt.approx`
  and the tensor core's k32 step are core's (`RsqrtApprox_v1`, `HopperE4m3QgmmaDot32_v1`, each pinned by its own vectors);
* against vLLM's torch references for the same kernels (tests/kernels/test_fused_deepseek_v32_norm_rope.py:77-112, 618-620 and
  tests/kernels/attention/test_deepgemm_attention.py:67-91 at d9105ea80) within the tolerance those tests allow: 1 fp8 step;
* the ue8m0 scale near every power of two and at its edge words (amax 0, subnormal, 1e-4, the largest finite, inf, NaN), where
  `__nv_log2f`'s ceiling is not torch's `ceil(log2)`;
* the short-row top-k against `i < len ? i : -1`;
* the logit against eight an H200 computed (`data/dsa_h200_logits.npz`): `DsaMqaLogit_v3`'s words, none of them v1's.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.prims import HopperE4m3QgmmaDot32 as DOT
from verity_vllm.program.registry import dsa as D
from verity_vllm.program.registry import prims as P
from tests.program.test_fp8_moe import _ffma, _fmul, _q, _rn

SIGN, INF, M32 = 1 << 31, 0x7F800000, 0xFFFFFFFF
NAN_PTX = 0x7FFFFFFF
NEG_INF = SIGN | INF


# --- the model -------------------------------------------------------------------------------------------------------------------

def _is_nan(w: int) -> bool:
    return (w >> 23) & 0xFF == 0xFF and bool(w & 0x7FFFFF)


def _ftz(w: int) -> int:
    return w & SIGN if (w >> 23) & 0xFF == 0 else w


def _fadd(a: int, b: int) -> int:
    """add.f32: the exact sum rounded once; an exact zero is -0 only when both operands are -0."""
    return _rn(_q(a) + _q(b), a & b & SIGN)


def _ffma_ftz(a: int, b: int, c: int) -> int:
    return _ftz(_ffma(_ftz(a), _ftz(b), _ftz(c)))


def _neg(w: int) -> int:
    return w ^ SIGN


def _i2f(e: int) -> int:
    """cvt.rn.f32.s32."""
    return _rn(Fraction(e - (1 << 32) if e >> 31 else e), 0)


def _val(w: int) -> float:
    return float(np.uint32(w).view(np.float32))


def _fmax(a: int, b: int) -> int:
    """max.f32: a NaN operand yields the other (two give 0x7FFFFFFF), max(+0, -0) = +0."""
    if _is_nan(a):
        return NAN_PTX if _is_nan(b) else b
    if _is_nan(b):
        return a
    if _val(a) == _val(b):
        return a & b
    return a if _val(a) > _val(b) else b


def _floor_log2(v: Fraction) -> int:
    e = v.numerator.bit_length() - v.denominator.bit_length()
    return e - (Fraction(2) ** e > v)


def log2f(s: int) -> int:
    """`__nv_log2f` as the PTX inlines it (fused_norm_rope.ptx:1282-1323), both branches kept."""
    pre = not _is_nan(s) and _val(s) < 2.0 ** -126                                         # setp.lt.f32 s, 2^-126
    s1 = _fmul(s, 0x4B000000) if pre and s & 0x7FFFFFFF != INF else s
    if s1 >= INF:                                                                         # setp.lt.u32 s1, 0x7F800000 fails
        return INF if s1 == INF else NAN_PTX                                              # fma.rn.ftz(s1, inf, inf)
    e = ((s1 - 0x3F3504F3) & M32) & 0xFF800000
    m = (s1 - e) & M32
    ef = _ffma_ftz(_i2f(e), 0x34000000, 0xC1B80000 if pre else 0)
    t = _fadd(m, 0xBF800000)
    p = _ffma_ftz(D.LOG2F_POLY[0], t, D.LOG2F_POLY[1])
    for c in D.LOG2F_POLY[2:]:
        p = _ffma_ftz(p, t, c)
    return _fadd(ef, _ffma_ftz(t, 0x3FB8AA3B, _fmul(t, _fmul(t, p))))


def _ceil_ftz(y: int) -> int:
    if _is_nan(y):
        return NAN_PTX
    if y & 0x7FFFFFFF == INF:
        return y
    v = _q(_ftz(y))
    c = math.ceil(v)
    return _rn(Fraction(c), y & SIGN)


def _ex2_integer(k: int) -> int:
    """ex2.approx.ftz on an integer word: 2^k exactly, +inf from 128, +0 below -126 (2^-127 is subnormal), 1 at -0."""
    if k & 0x7FFFFFFF == INF:
        return 0 if k & SIGN else INF
    v = _q(k)
    assert v.denominator == 1, hex(k)
    n = int(v)
    return INF if n >= 128 else 0 if n < -126 else (n + 127) << 23


def ue8m0_scale(amax: int) -> int:
    """kernels.py:57-58: `exp2(ceil(log2(div_rn(maximum(amax, 1e-4), 448))))`."""
    s = _fmax(amax, D.F_1EM4)
    s = INF if s == INF else _rn(_q(s) / 448, 0)
    return _ex2_integer(_ceil_ftz(log2f(s)))


def _e4m3(w: int) -> int:
    """cvt.rn.satfinite.e4m3x2.f32 from the exact value: NaN 0x7F, past 448 (infinities too) the largest finite of the sign."""
    if _is_nan(w):
        return 0x7F
    s = 0x80 if w & SIGN else 0
    if w & 0x7FFFFFFF == INF:
        return s | 0x7E
    v = abs(_q(w))
    if v == 0:
        return s
    unit = Fraction(2) ** (max(_floor_log2(v), -6) - 3)
    n, r = divmod(v / unit, 1)
    n += r > Fraction(1, 2) or (r == Fraction(1, 2) and n & 1)
    val = min(n * unit, Fraction(448))
    if val <= Fraction(1, 64):
        return s | int(val * 512)
    e = _floor_log2(val)
    return s | (e + 7) << 3 | int(val / Fraction(2) ** (e - 3)) - 8


def _div_by_scale(x: int, scale: int) -> int:
    if _is_nan(x):
        return NAN_PTX
    if scale == INF:
        return NAN_PTX if x & 0x7FFFFFFF == INF else x & SIGN
    if x & 0x7FFFFFFF == INF:
        return x
    return _rn(_q(x) / _q(scale), x & SIGN)


def quant(x: list[int]) -> tuple[list[int], int]:
    """`_fp8_ue8m0_quantize` (kernels.py:50-60): amax over |x| (max.f32 drops NaNs), the scale, each byte."""
    amax = x[0] & 0x7FFFFFFF
    for v in x[1:]:
        amax = _fmax(amax, v & 0x7FFFFFFF)
    scale = ue8m0_scale(amax)
    return [_e4m3(_div_by_scale(v, scale)) for v in x], scale


def _butterfly(lanes: list[int], op, offsets) -> int:
    """`shfl.sync.bfly` at each offset in turn, lane l becoming op(own, partner); lane 0's word."""
    v = list(lanes)
    for o in offsets:
        v = [op(v[l], v[l ^ o]) for l in range(len(v))]
    return v[0]


def _block_sum(lanes: list[int], first=None) -> int:
    """`tl.sum` over the thread-per-element row: each warp's butterfly (the first level ``first`` when given), then the warps'
    words through shared memory and a butterfly over them."""
    warps = []
    for g in range(len(lanes) // 32):
        w = lanes[32 * g:32 * g + 32]
        if first is not None:
            w = [first(w[l], w[l ^ 16]) for l in range(32)]
            warps.append(_butterfly(w, _fadd, (8, 4, 2, 1)))
        else:
            warps.append(_butterfly(w, _fadd, (16, 8, 4, 2, 1)))
    return _butterfly(warps, _fadd, [o for o in (2, 1) if o < len(warps)])


def _rope(x: list[int], cs: list[int], rope: int) -> list[int]:
    """The interleaved rotation (kernels.py:349-393): pair j's even element `fma(x0, cos, mul(-x1, sin))`, its odd one
    `fma(x1, cos, mul(x0, sin))`."""
    half = rope // 2
    out = list(x)
    for j in range(half):
        c, s = cs[j] << 16, cs[half + j] << 16
        x0, x1 = x[2 * j], x[2 * j + 1]
        out[2 * j] = _ffma(x0, c, _fmul(_neg(x1), s))
        out[2 * j + 1] = _ffma(x1, c, _fmul(x0, s))
    return out


def k_row(k: list[int], w: list[int], b: list[int], cs: list[int], eps: float, rope: int = 64) -> tuple[list[int], int]:
    """`_fused_norm_rope_kernel` program 0 (kernels.py:322-415): the LayerNorm, the RoPE, the quantisation."""
    x = [v << 16 for v in k]
    n = len(x)
    inv = D.f32_bits(1.0 / n)
    mean = _fmul(_block_sum(x), inv)
    d = [_fadd(v, _neg(mean)) for v in x]
    var = _fmul(_block_sum(d, first=lambda own, partner: _ffma(own, own, _fmul(partner, partner))), inv)
    rstd = evaluate(P.RsqrtApprox, _fadd(D.f32_bits(eps), var))[0]
    normed = [_ffma(_fmul(d[i], rstd), w[i], b[i]) for i in range(n)]
    return quant(_rope(normed, cs, rope))


def q_head(q: list[int], cs: list[int], w: int, nh: int, rope: int = 64) -> tuple[list[int], int]:
    """`_fused_q_kernel` program 1 (kernels.py:770-826): the RoPE, the quantisation, the weight times the scale, 128^-0.5 and
    nh^-0.5."""
    qb, scale = quant(_rope([v << 16 for v in q], cs, rope))
    weight = _fmul(D.f32_bits(nh ** -0.5), _fmul(D.f32_bits(len(q) ** -0.5), _fmul(scale, w << 16)))
    return qb, weight


def logit(q: list[list[int]], w: list[int], k: list[int], ks: int, contracted: bool = True) -> int:
    """DeepGEMM's sm90 MQA logits epilogue for one (token, key) (sm90_fp8_mqa_logits.cuh:270-302): the m64nN accumulator gives
    quad lane c heads 8j + 2c + {0, 1}; each lane's two sums of relu * w, `(s0 + s1) * ks`, and the quad's xor-1 then xor-2
    shuffle adds.  A sum is FFMAs in ascending j from j = 0's product, which nvcc contracts into j = 1's add (``contracted``, the
    SASS: `fma(r0, w0, r1 * w1)`) or which is its own FMUL (the source as written)."""
    H, Dd = len(q), len(k)
    relu = []
    for h in range(H):
        acc = 0
        for s in range(0, Dd, 32):
            acc = DOT.evaluate(acc, *k[s:s + 32], *q[h][s:s + 32])
        relu.append(_fmax(acc, 0))
    lane = []
    for c in range(4):
        sums = []
        for p in (0, 1):
            heads = [8 * j + 2 * c + p for j in range(H // 8)]
            if contracted and len(heads) > 1:
                s, rest = _ffma(relu[heads[0]], w[heads[0]], _fmul(relu[heads[1]], w[heads[1]])), heads[2:]
            else:
                s, rest = _fmul(relu[heads[0]], w[heads[0]]), heads[1:]
            for h in rest:
                s = _ffma(relu[h], w[h], s)
            sums.append(s)
        lane.append(_fmul(_fadd(sums[0], sums[1]), ks))
    return _butterfly(lane, _fadd, (1, 2))


# --- inputs ----------------------------------------------------------------------------------------------------------------------

def _bf16(rng, n: int, scale: float = 1.0) -> list[int]:
    return ((rng.standard_normal(n).astype(np.float32) * np.float32(scale)).view(np.uint32) >> 16).astype(int).tolist()


def _f32(rng, n: int, mean: float, std: float) -> list[int]:
    return (rng.normal(mean, std, n).astype(np.float32)).view(np.uint32).astype(int).tolist()


def _cs(rng, rope: int = 64) -> list[int]:
    """A rotary cache row in bf16, as `get_rope` builds it in the default dtype: [cos | sin] at one position."""
    half = rope // 2
    pos = int(rng.integers(0, 8192))
    theta = pos / 10000.0 ** (np.arange(half) / half)
    v = np.concatenate([np.cos(theta), np.sin(theta)]).astype(np.float32)
    return ((v.view(np.uint32) + 0x7FFF + ((v.view(np.uint32) >> 16) & 1)) >> 16).astype(int).tolist()


def _flat(out) -> list[int]:
    return [int(v) for v in out]


KROW = bind(D.DsaIndexerKRow, D=128, ROPE=64, EPS=1e-6)
QHEAD = bind(D.DsaIndexerQHead, D=128, ROPE=64, NH=32)
LOGIT = bind(D.DsaMqaLogitV3, H=32, D=128)
LOGIT_V1 = bind(D.DsaMqaLogit, H=32, D=128)
#: eight (token, key) logits of an H200's DeepGEMM sm90 kernels (property glm53_dsa_capture, run r20261012-222325-ce92), four
#: prefill and four paged (``paged``), each the first ones of its kind where `DsaMqaLogit_v1` differs: q (8, 32, 128) and k (8, 128)
#: e4m3 bytes, w (8, 32) and ks (8,) f32 words, the kernel's logit (8,)
H200_LOGITS = Path(__file__).parent / "data" / "dsa_h200_logits.npz"


# --- the scale ---------------------------------------------------------------------------------------------------------------

def _scale_cases() -> list[int]:
    """amax words: the edges, and around 448 * 2^k for every k the scale reaches (where `ceil(log2)` turns)."""
    out = [0, SIGN, 1, 0x007FFFFF, 0x00800000, D.F_1EM4 - 1, D.F_1EM4, D.F_1EM4 + 1, 0x7F7FFFFF, INF, 0x7FC00000, NAN_PTX,
           0xBF800000, NEG_INF]
    for k in range(-22, 120):
        c = D.f32_bits(448.0 * 2.0 ** k)
        out += [c + d for d in (-2, -1, 0, 1, 2, 3, 0x40, 0x400)]
    return out


def test_scale_matches_the_log2f_model() -> None:
    cases = _scale_cases()
    for a in cases:
        assert evaluate(D.DsaUe8m0Scale, a)[0] == ue8m0_scale(a), hex(a)


def test_scale_is_a_power_of_two_or_inf() -> None:
    for a in _scale_cases():
        s = ue8m0_scale(a)
        assert s == INF or (s & 0x7FFFFF == 0 and 0x34800000 <= s <= 0x7F000000), (hex(a), hex(s))


def test_log2f_ceiling_differs_from_torch_just_above_a_power_of_two() -> None:
    """`__nv_log2f(2^k (1 + 2^-23))` rounds to k for most k, so the kernel's scale is 2^k where `torch.log2` (vLLM's reference,
    test_fused_deepseek_v32_norm_rope.py:104-110) gives 2^(k + 1): the GPU then saturates the amax element at 448.  The scale
    and the bytes here are the kernel's."""
    diff = []
    for k in range(-21, 120):
        s = D.f32_bits(2.0 ** k) + 1
        exact = k + 1
        y = log2f(s)
        if math.ceil(_q(y)) != exact:
            diff.append(k)
    assert diff and 0 not in diff and -1 not in diff


def test_mufu_ex2_on_the_integers_the_ceiling_gives() -> None:
    for n in range(-126, 129):
        assert evaluate(P.MufuEx2Ftz, _rn(Fraction(n), 0))[0] == _ex2_integer(_rn(Fraction(n), 0)), n
    assert evaluate(P.MufuEx2Ftz, SIGN)[0] == 0x3F800000


def test_ceil_primitive_against_math_ceil() -> None:
    rng = np.random.default_rng(1)
    words = [0, SIGN, 1, SIGN | 1, 0x3F000000, 0xBF000000, 0x3F800001, 0xBF800001, 0x4B000000, 0x4AFFFFFF, 0xCAFFFFFF, 0x7F7FFFFF,
             INF, NEG_INF, 0x7FC00000, 0xFFC00001, 0x3EFFFFFF, 0xBEFFFFFF]
    words += rng.integers(0, 1 << 32, 4000, dtype=np.uint64).astype(int).tolist()
    words += (rng.standard_normal(2000).astype(np.float32) * 100).view(np.uint32).astype(int).tolist()
    for w in words:
        assert evaluate(D.F32CeilFtz, w)[0] == _ceil_ftz(w), hex(w)


def test_integer_primitives() -> None:
    rng = np.random.default_rng(2)
    for a, b in rng.integers(0, 1 << 32, (2000, 2), dtype=np.uint64).astype(int).tolist():
        assert evaluate(D.U32And, a, b)[0] == a & b
        assert evaluate(D.I32Sub, a, b)[0] == (a - b) % (1 << 32)


# --- the quantisation, the rows ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N", [4, 32, 128])
def test_quant_matches_the_model(N: int) -> None:
    rng = np.random.default_rng(N)
    rows = [_f32(rng, N, 0.0, float(s)) for s in (1e-3, 0.3, 7.0, 300.0, 1e5)]
    special = _f32(rng, N, 0.0, 1.0)
    special[1], special[N - 1] = 0x7FC00000, SIGN
    rows += [special, [0] * N, [0x7FC00000] * N, [INF] + [0x3F800000] * (N - 1), [0x00000001] * N]
    q = bind(D.DsaUe8m0Quant, N=N)
    for x in rows:
        got = _flat(evaluate(q, x))
        qb, s = quant(x)
        assert got == qb + [s]


def _k_case(rng, scale: float):
    return _bf16(rng, 128, scale), _f32(rng, 128, 1.0, 0.2), _f32(rng, 128, 0.0, 0.1), _cs(rng)


def test_k_row_matches_the_kernel_model() -> None:
    rng = np.random.default_rng(3)
    cases = [_k_case(rng, s) for s in (0.01, 1.0, 1.0, 1.0, 30.0, 1e3)]
    k, w, b, cs = _k_case(rng, 1.0)
    cases.append(([0] * 128, w, b, cs))                                                   # var 0: rstd = rsqrt(eps)
    cases.append(([0x3F80] * 64 + [0xBF80] * 64, w, b, cs))
    for k, w, b, cs in cases:
        qb, s = k_row(k, w, b, cs, 1e-6)
        assert _flat(evaluate(KROW, k, w, b, cs)) == qb + [s]


@pytest.mark.parametrize("Dd", [32, 64])
def test_k_row_at_fewer_warps_matches_the_model(Dd: int) -> None:
    rng = np.random.default_rng(Dd)
    fn = bind(D.DsaIndexerKRow, D=Dd, ROPE=16, EPS=1e-6)
    for _ in range(3):
        k, w, b, cs = _bf16(rng, Dd), _f32(rng, Dd, 1.0, 0.2), _f32(rng, Dd, 0.0, 0.1), _cs(rng, 16)
        qb, s = k_row(k, w, b, cs, 1e-6, rope=16)
        assert _flat(evaluate(fn, k, w, b, cs)) == qb + [s]


def test_q_head_matches_the_kernel_model() -> None:
    rng = np.random.default_rng(4)
    for scale in (0.01, 1.0, 1.0, 50.0):
        q, cs, w = _bf16(rng, 128, scale), _cs(rng), _bf16(rng, 1)[0]
        qb, weight = q_head(q, cs, w, 32)
        assert _flat(evaluate(QHEAD, q, cs, w)) == qb + [weight]


def test_q_token_is_its_heads() -> None:
    rng = np.random.default_rng(5)
    H = 8
    q, cs, w = [_bf16(rng, 128) for _ in range(H)], _cs(rng), _bf16(rng, H)
    got = _flat(evaluate(bind(D.DsaIndexerQ, H=H, D=128, ROPE=64), [v for r in q for v in r], cs, w))
    heads = [q_head(q[h], cs, w[h], H) for h in range(H)]
    assert got == [v for qb, _ in heads for v in qb] + [wt for _, wt in heads]


def _e4m3_row(rng, n: int) -> list[int]:
    v = rng.integers(0, 0x7F, n)
    return (v | (rng.integers(0, 2, n) << 7)).astype(int).tolist()


@pytest.mark.parametrize("fn,contracted", [(LOGIT, True), (LOGIT_V1, False)], ids=["v3", "v1"])
def test_logit_matches_the_epilogue_model(fn, contracted: bool) -> None:
    rng = np.random.default_rng(6)
    for _ in range(6):
        q = [_e4m3_row(rng, 128) for _ in range(32)]
        w, k, ks = _f32(rng, 32, 0.0, 0.05), _e4m3_row(rng, 128), D.f32_bits(2.0 ** int(rng.integers(-12, 4)))
        assert evaluate(fn, [v for r in q for v in r], w, k, ks)[0] == logit(q, w, k, ks, contracted)
    q = [[0x7F] + _e4m3_row(rng, 127) for _ in range(32)]                                  # NaN products: relu drops them
    w, k = _f32(rng, 32, 0.0, 0.05), _e4m3_row(rng, 128)
    assert evaluate(fn, [v for r in q for v in r], w, k, 0x3F800000)[0] == logit(q, w, k, 0x3F800000, contracted)


def test_the_h200s_logits_are_v3s_and_refute_v1() -> None:
    """Every kept logit is `DsaMqaLogit_v3`'s word and none is v1's: the H200's SASS contracts each sum's first product."""
    z = np.load(H200_LOGITS)
    assert sorted(z["paged"].tolist()) == [0] * 4 + [1] * 4
    for q, w, k, ks, want in zip(z["q"], z["w"], z["k"], z["ks"], z["logit"]):
        args = (q.reshape(-1).tolist(), w.tolist(), k.tolist(), int(ks))
        assert evaluate(LOGIT, *args)[0] == int(want)
        assert evaluate(LOGIT_V1, *args)[0] != int(want)
        assert logit(q.tolist(), w.tolist(), k.tolist(), int(ks)) == int(want)


def test_at_one_head_per_sum_v3_is_v1() -> None:
    """At H = 8 each lane's sum has one product, so there is nothing to contract: v3 and v1 are the same function."""
    rng = np.random.default_rng(12)
    v1, v3 = bind(D.DsaMqaLogit, H=8, D=64), bind(D.DsaMqaLogitV3, H=8, D=64)
    for _ in range(4):
        q, w, k, ks = [_e4m3_row(rng, 64) for _ in range(8)], _f32(rng, 8, 0.0, 0.05), _e4m3_row(rng, 64), D.f32_bits(0.25)
        args = ([v for r in q for v in r], w, k, ks)
        assert evaluate(v3, *args) == evaluate(v1, *args)


@pytest.mark.parametrize("fn,contracted", [(D.DsaMqaLogitsV3, True), (D.DsaMqaLogits, False)], ids=["v3", "v1"])
def test_logits_are_the_logit_per_key(fn, contracted: bool) -> None:
    rng = np.random.default_rng(7)
    N = 3
    q = [_e4m3_row(rng, 128) for _ in range(32)]
    w = _f32(rng, 32, 0.0, 0.05)
    ks = [_e4m3_row(rng, 128) for _ in range(N)]
    sc = [D.f32_bits(0.5), D.f32_bits(0.125), D.f32_bits(4.0)]
    got = _flat(evaluate(bind(fn, N=N, H=32, D=128), [v for r in q for v in r], w, [v for r in ks for v in r], sc))
    assert got == [logit(q, w, ks[n], sc[n], contracted) for n in range(N)]


@pytest.mark.parametrize("TOPK", [16, 2048])
def test_short_row_topk(TOPK: int) -> None:
    fn = bind(D.DsaTopKShortRow, TOPK=TOPK)
    for n in (0, 1, 5, TOPK - 1, TOPK, (-3) & M32, (1 << 31) - 1, 1 << 31):
        signed = n - (1 << 32) if n >> 31 else n
        assert _flat(evaluate(fn, n)) == [i if i < signed else M32 for i in range(TOPK)], n


# --- vLLM's torch references ---------------------------------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def _ref_layer_norm(x, w, b, eps):
    """test_fused_deepseek_v32_norm_rope.py:77-81."""
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    return (xf - mean) * torch.rsqrt(var + eps) * w.float() + b.float()


def _ref_rope(x, cos_sin):
    """test_fused_deepseek_v32_norm_rope.py:84-102, interleaved, at one position."""
    rot = cos_sin.shape[-1]
    half = rot // 2
    cos, sin = cos_sin[:half], cos_sin[half:]
    out = x.float().clone()
    r = out[..., :rot]
    x1, x2 = r[..., 0::2].clone(), r[..., 1::2].clone()
    r[..., 0::2] = x1 * cos - x2 * sin
    r[..., 1::2] = x2 * cos + x1 * sin
    return out


def _ref_quant(vals):
    """test_fused_deepseek_v32_norm_rope.py:104-110."""
    amax = vals.float().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax, min=1e-4) / 448.0
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return (vals.float() / scale).to(torch.float8_e4m3fn), scale.squeeze(-1)


def _fp8_steps(a: list[int], b: list[int]) -> int:
    """test_fused_deepseek_v32_norm_rope.py:122-127: the largest distance in representable fp8 steps."""
    key = lambda u: 0xFF - u if u >= 0x80 else u + 0x80  # noqa: E731
    return max(abs(key(x) - key(y)) for x, y in zip(a, b))


def _t_bf16(words):
    return torch.tensor(np.asarray(words, np.uint16).view(np.int16)).view(torch.bfloat16)


def _t_f32(words):
    return torch.tensor(np.asarray(words, np.uint32).view(np.float32))


def _bytes(t) -> list[int]:
    return t.view(torch.uint8).tolist()


def test_k_row_within_vllm_reference_tolerance() -> None:
    """vLLM's own test allows 1 fp8 step on the bytes (`assert_fp8`); the scale is the same word."""
    rng = np.random.default_rng(8)
    for scale in (0.01, 1.0, 30.0):
        k, w, b, cs = _k_case(rng, scale)
        got = _flat(evaluate(KROW, k, w, b, cs))
        ref = _ref_rope(_ref_layer_norm(_t_bf16(k), _t_f32(w), _t_f32(b), 1e-6), _t_bf16(cs).float())
        qb, s = _ref_quant(ref)
        assert _fp8_steps(got[:128], _bytes(qb)) <= 1
        assert got[128] == int(s.view(torch.int32).item()) & M32


def test_q_head_within_vllm_reference_tolerance() -> None:
    """test_fused_deepseek_v32_norm_rope.py:612-620: the bytes within 1 fp8 step, the weight `w * scale * 128^-0.5 * 32^-0.5`
    within an f32 rounding of each product."""
    rng = np.random.default_rng(9)
    for scale in (0.01, 1.0, 50.0):
        q, cs, w = _bf16(rng, 128, scale), _cs(rng), _bf16(rng, 1)[0]
        got = _flat(evaluate(QHEAD, q, cs, w))
        qb, s = _ref_quant(_ref_rope(_t_bf16(q).float(), _t_bf16(cs).float()))
        weight = (_t_bf16([w]).float() * s * (128 ** -0.5) * (32 ** -0.5)).item()
        assert _fp8_steps(got[:128], _bytes(qb)) <= 1
        assert math.isclose(float(np.uint32(got[128]).view(np.float32)), weight, rel_tol=4e-7)


def test_logit_against_vllm_reference() -> None:
    """test_deepgemm_attention.py:67-91 (`(score.relu() * weights).sum` over heads) with the key's scale applied, as
    `fp8_mqa_logits` applies it (sparse_attn_indexer.py:496-503), within that test's bound: `calc_diff` (vllm/utils/deep_gemm.py:743-
    756) below 1e-3 (test_deepgemm_attention.py:150-151).  The k32 step keeps 14 bits a step, so the words are not torch's."""
    rng = np.random.default_rng(10)
    N = 8
    q = [_e4m3_row(rng, 128) for _ in range(32)]
    w = _f32(rng, 32, 0.0, 0.05)
    ks = [_e4m3_row(rng, 128) for _ in range(N)]
    sc = [D.f32_bits(2.0 ** -int(rng.integers(0, 6))) for _ in range(N)]
    got = np.asarray(_flat(evaluate(bind(D.DsaMqaLogitsV3, N=N, H=32, D=128), [v for r in q for v in r], w,
                                    [v for r in ks for v in r], sc)), np.uint32).view(np.float32).astype(np.float64)
    qt = torch.tensor(q, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    kt = torch.tensor(ks, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    score = torch.einsum("hd,nd->hn", qt, kt)
    ref = ((score.relu() * _t_f32(w).double()[:, None]).sum(dim=0) * _t_f32(sc).double()).numpy()
    diff = 1 - 2 * float((got * ref).sum()) / float((got * got + ref * ref).sum())
    assert diff < 1e-3
