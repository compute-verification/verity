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
* the long-row top-k: `DsaTopKSelection_v1` against its twin, its Boolean v2 against it, and the threshold form v3 and its chunked
  Boolean v4 against v1 at every threshold class, on every row of special words at the smallest statics, on the capture's kinds
  of row and the kernels' overflow, and at the served statics on the capture's own rows (``VERITY_VLLM_DSA_SERVED``).
"""
from __future__ import annotations

import math
import os
from fractions import Fraction

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


def logit(q: list[list[int]], w: list[int], k: list[int], ks: int) -> int:
    """DeepGEMM's sm90 MQA logits epilogue for one (token, key) (sm90_fp8_mqa_logits.cuh:270-302): the m64nN accumulator gives
    quad lane c heads 8j + 2c + {0, 1}; each lane's two sums of relu * w by FFMA in ascending j, `(s0 + s1) * ks`, and the
    quad's xor-1 then xor-2 shuffle adds."""
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
            s = _fmul(relu[heads[0]], w[heads[0]])
            for h in heads[1:]:
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
LOGIT = bind(D.DsaMqaLogit, H=32, D=128)


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


def test_logit_matches_the_epilogue_model() -> None:
    rng = np.random.default_rng(6)
    for _ in range(6):
        q = [_e4m3_row(rng, 128) for _ in range(32)]
        w, k, ks = _f32(rng, 32, 0.0, 0.05), _e4m3_row(rng, 128), D.f32_bits(2.0 ** int(rng.integers(-12, 4)))
        assert evaluate(LOGIT, [v for r in q for v in r], w, k, ks)[0] == logit(q, w, k, ks)
    q = [[0x7F] + _e4m3_row(rng, 127) for _ in range(32)]                                  # NaN products: relu drops them
    w, k = _f32(rng, 32, 0.0, 0.05), _e4m3_row(rng, 128)
    assert evaluate(LOGIT, [v for r in q for v in r], w, k, 0x3F800000)[0] == logit(q, w, k, 0x3F800000)


def test_logits_are_the_logit_per_key() -> None:
    rng = np.random.default_rng(7)
    N = 3
    q = [_e4m3_row(rng, 128) for _ in range(32)]
    w = _f32(rng, 32, 0.0, 0.05)
    ks = [_e4m3_row(rng, 128) for _ in range(N)]
    sc = [D.f32_bits(0.5), D.f32_bits(0.125), D.f32_bits(4.0)]
    got = _flat(evaluate(bind(D.DsaMqaLogits, N=N, H=32, D=128), [v for r in q for v in r], w, [v for r in ks for v in r], sc))
    assert got == [logit(q, w, ks[n], sc[n]) for n in range(N)]


@pytest.mark.parametrize("TOPK", [16, 2048])
def test_short_row_topk(TOPK: int) -> None:
    fn = bind(D.DsaTopKShortRow, TOPK=TOPK)
    for n in (0, 1, 5, TOPK - 1, TOPK, (-3) & M32, (1 << 31) - 1, 1 << 31):
        signed = n - (1 << 32) if n >> 31 else n
        assert _flat(evaluate(fn, n)) == [i if i < signed else M32 for i in range(TOPK)], n


def _top(words: np.ndarray, n: int, T: int, rng) -> list[int]:
    """A top-T set of words[:n] under the kernels' order, in a random order."""
    return [int(i) for i in rng.permutation(np.argsort(-D.order_keys(words[:n]), kind="stable")[:T])]


def _selection_cases(T: int, N: int, seed: int) -> tuple[list, list[bool]]:
    """(words, len, idx) cases and whether each is a selection: the shortcut's words and near misses at lengths up to TOPK
    (negative included); past it a random row's top-T set in any order, the same with a duplicate, an index at len, -1, past N
    or 2^31, a key below an unselected one in, a row of equal words (any T distinct; a duplicate, or an index past the row or
    equal to a free position's low bits, fails though no key tells), +0 taken over -0 and not, +NaN on top and -NaN at the
    bottom; len > N."""
    rng = np.random.default_rng(seed)
    cases, want = [], []

    def add(words, n, idx, ok):
        cases.append((np.asarray(words, np.uint32), n & M32, [i & M32 for i in idx]))
        want.append(ok)

    rnd = lambda: rng.standard_normal(N).astype(np.float32).view(np.uint32)  # noqa: E731
    for n in (0, 1, T - 1, T, -3, -(1 << 31)):
        short = [i if i < n else -1 for i in range(T)]
        add(rnd(), n, short, True)
        add(rnd(), n, short[:-1] + [short[-1] + 1], False)
        add(rnd(), n, list(reversed(short)), T == 1 or max(n, 0) <= 0)
    for n in sorted({n for n in (T + 1, T + 2, N - 1, N) if T < n <= N}):
        w = rnd()
        top = _top(w, n, T, rng)
        add(w, n, top, True)
        add(w, n, sorted(top), True)
        add(w, n, top[:-1] + [top[0]], T == 1)
        for bad in (n, -1, N + 5, 1 << 31):
            add(w, n, top[:-1] + [bad], False)
        rest = [j for j in np.argsort(-D.order_keys(w[:n]), kind="stable") if j not in top]
        add(w, n, top[:-1] + [int(rest[-1])], False)
        add(w, n, [int(i) for i in rng.choice(n, T, replace=False)], None)
        eq = np.full(N, D.f32_bits(1.5), np.uint32)
        pick = [int(i) for i in rng.choice(n, T, replace=False)]
        add(eq, n, pick, True)
        add(eq, n, pick[:-1] + [pick[0]], T == 1)
        free = next(j for j in range(n) if j not in pick)
        for bad in (free | 1 << 31, free + (1 << 20), n, -1):
            add(eq, n, pick[:-1] + [bad], False)
        z = np.full(N, D.f32_bits(-1.0), np.uint32)
        z[:T - 1] = D.f32_bits(1.0)
        z[n - 1], z[n - 2] = 0, SIGN
        add(z, n, list(range(T - 1)) + [n - 1], True)
        add(z, n, list(range(T - 1)) + [n - 2], False)
        nan = rnd()
        nan[n - 1], nan[0] = 0x7FC00000, 0xFFC00000
        top = _top(nan, n, T, rng)
        add(nan, n, top, True)
        assert n - 1 in top and 0 not in top
    w = rnd()
    add(w, N + 1, _top(w, N, T, rng), False)
    add(w, (1 << 31) - 1, _top(w, N, T, rng), False)
    return cases, want


def _arrays(cases) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.stack([c[0] for c in cases]), np.array([c[1] for c in cases], np.uint32),
            np.array([c[2] for c in cases], np.uint32))


@pytest.mark.parametrize("T, N", [(1, 2), (4, 9), (16, 64), (5, 37)])
def test_topk_selection_against_its_twin(T: int, N: int) -> None:
    """`DsaTopKSelection_v1` against the numpy twin `selection_ok` and the cases' own verdicts, and the IR oracle on a sample."""
    from verity.ml.circuits import evaluation as E

    fn = bind(D.DsaTopKSelection, TOPK=T, N=N)
    cases, want = _selection_cases(T, N, T * 1000 + N)
    got = np.asarray(E.evaluate_batch(fn, *_arrays(cases))).reshape(-1).tolist()
    twin = [D.selection_ok(w, n, idx) for w, n, idx in cases]
    assert got == [int(t) for t in twin]
    assert all(t == ok for t, ok in zip(twin, want) if ok is not None)
    for c in cases[::7]:
        assert evaluate(fn, [int(v) for v in c[0]], c[1], c[2])[0] == D.selection_ok(*c)


def test_topk_selection_twin_at_dsv4_width() -> None:
    """The twin at top-1024 (DSV4's indexer) on a row of 4096 keys with 3000 tied at the threshold: any 1024 of the row's top set
    pass, one key below the threshold swapped in fails."""
    rng = np.random.default_rng(12)
    w = np.full(4096, D.f32_bits(0.25), np.uint32)
    w[rng.choice(4096, 1000, replace=False)] = D.f32_bits(-2.0)
    w[3000:3100] = D.f32_bits(3.0)
    tied = np.flatnonzero(w == D.f32_bits(0.25))
    pick = list(range(3000, 3100)) + [int(i) for i in rng.choice(tied, 924, replace=False)]
    assert D.selection_ok(w, 4096, rng.permutation(pick))
    low = int(np.flatnonzero(w == D.f32_bits(-2.0))[0])
    assert not D.selection_ok(w, 4096, pick[:-1] + [low])
    assert not D.selection_ok(w, 4096, pick[:-1] + [pick[0]])


@pytest.mark.parametrize("T, N", [(1, 2), (4, 9), (16, 64), (5, 37)])
def test_boolean_topk_selection_is_its_word_view(T: int, N: int) -> None:
    """`DsaTopKSelection_v2` bit for bit against v1 on the selection cases and on random words, lengths and indices; the switch
    maps v1 to it."""
    from verity.ml.circuits import evaluation as E
    from verity.ml.circuits.evaluation import bits as BS
    from verity.ml.circuits.boolean import trace as TR
    from verity_vllm.program import boolean as SW
    from verity_vllm.program.registry import boolean_dsa as BD

    v2 = bind(BD.DsaTopKSelectionV2, TOPK=T, N=N)
    word = TR.word_of(v2)
    assert word.id == bind(D.DsaTopKSelection, TOPK=T, N=N).id
    assert SW.boolean_version(word).id == v2.id
    cases, _ = _selection_cases(T, N, T * 1000 + N + 1)
    rng = np.random.default_rng(N)
    m = 256
    noise = (rng.integers(0, 1 << 32, (m, N), dtype=np.uint64).astype(np.uint32),
             rng.integers(-2, N + 3, m).astype(np.int64).astype(np.uint32),
             rng.integers(-1, N + 1, (m, T)).astype(np.int64).astype(np.uint32))
    for args in (_arrays(cases), noise):
        got = np.asarray(BS.evaluate_words(v2, *args)).reshape(-1)
        assert np.array_equal(got, np.asarray(E.evaluate_batch(word, *args)).reshape(-1))


# --- DsaTopKSelection_v3 and _v4: the threshold witness --------------------------------------------------------------------------

#: the f32 words of the exhaustive rows: both zeros, +-1, both infinities, PTX's NaN and a negative quiet NaN
_SPECIAL = (0, SIGN, D.f32_bits(1.0), D.f32_bits(-1.0), INF, NEG_INF, NAN_PTX, 0xFFC00000)
_BATCH = 1 << 16


def _key_word(k: int) -> int:
    """The f32 word whose `order_key` is k."""
    return k if k >= 0 else D.INT_MAX - k


def _threshold_classes(words) -> list[int]:
    """A threshold word of each class the row's keys tell apart: at each key, just below and just above it, and at both ends."""
    ks = {int(k) + d for k in D.order_keys(words) for d in (-1, 0, 1)} | {-(1 << 31), (1 << 31) - 1}
    return sorted(_key_word(k) for k in ks if -(1 << 31) <= k < 1 << 31)


def _bit(out) -> np.ndarray:
    return np.asarray(out).reshape(-1).astype(np.int64)


def _i32(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.int64)
    return v - ((v >> 31) << 32)


def _selections(w: np.ndarray, n: np.ndarray, idx: np.ndarray, thr: np.ndarray | None = None) -> np.ndarray:
    """v1 (``thr`` None) or v3 on rows of arrays, from their definitions: the shortcut's words up to TOPK, past it TOPK distinct
    indices in [0, len) with len <= N, and then v1's least member key at least the greatest other key below len, or v3's members'
    keys at least the threshold's and the others' below len at most it."""
    N, T = w.shape[1], idx.shape[1]
    s, ix, j = _i32(n)[:, None], _i32(idx), np.arange(N)
    short = (ix == np.where(np.arange(T) < s, np.arange(T), -1)).all(1)
    member = (ix[:, :, None] == j).any(1)
    tight = ((ix >= 0) & (ix < s)).all(1) & (member.sum(1) == T) & (s[:, 0] <= N)
    key, live = D.order_keys(w), j < s
    if thr is None:
        fits = np.where(member, key, 1 << 32).min(1) >= np.where(~member & live, key, -(1 << 32)).max(1)
    else:
        t = D.order_keys(thr)[:, None]
        fits = ((~member | (key >= t)) & (member | ~live | (key <= t))).all(1)
    return np.where(s[:, 0] <= T, short, tight & fits).astype(np.int64)


def _threshold_agrees(T: int, N: int, Cs, cases, ir: int = 48, twins: int = 2000) -> np.ndarray:
    """v1 on ``cases`` is 1 exactly when v3 is 1 at some threshold class, and is v3 at `selection_threshold`; v4 at each chunk
    width of ``Cs`` is v3 at every class (a row of len <= TOPK or past N at two thresholds, 0 and all ones).  v1 and v3 are
    `_selections` here, checked against the IR oracle on ``ir`` instances spread over them and against the twins
    (`selection_ok`, `selection_ok_thr`, and `boolean_dsa`'s `chunk_ok` and `final_ok` composed) on ``twins``.  Returns v1."""
    from verity.ml.circuits import evaluation as E
    from verity.ml.circuits.evaluation import bits as BS
    from verity_vllm.program.registry import boolean_dsa as BD

    w, n, idx = _arrays(cases)
    signed = _i32(n)
    seen: dict[bytes, list[int]] = {}
    cls = []
    for i in range(len(n)):
        if T < signed[i] <= N:
            if (k := w[i].tobytes()) not in seen:
                seen[k] = _threshold_classes(w[i])
            cls.append(seen[k])
        else:
            cls.append([0, M32])
    rep = np.repeat(np.arange(len(n)), [len(c) for c in cls])
    thr = np.array([t for c in cls for t in c], np.uint32)
    got1, got3 = _selections(w, n, idx), np.empty(len(rep), np.int64)
    for lo in range(0, len(rep), _BATCH):
        r = rep[lo:lo + _BATCH]
        args = (w[r], n[r], idx[r], thr[lo:lo + _BATCH])
        got3[lo:lo + _BATCH] = _selections(*args)
        for C in Cs:
            fn = bind(BD.DsaTopKSelectionV4, TOPK=T, N=N, C=C)
            assert np.array_equal(_bit(BS.evaluate_words(fn, *args)), got3[lo:lo + _BATCH]), (fn.id, lo)
    assert np.array_equal(np.bincount(rep, got3, len(n)) > 0, got1 == 1)
    best = np.array([D.selection_threshold(*c) for c in cases], np.uint32)
    assert np.array_equal(_selections(w, n, idx, best), got1)
    at = np.unique(np.linspace(0, len(rep) - 1, ir).astype(np.int64))
    v1, v3 = bind(D.DsaTopKSelection, TOPK=T, N=N), bind(D.DsaTopKSelectionV3, TOPK=T, N=N)
    assert np.array_equal(_bit(E.evaluate_batch(v3, w[rep[at]], n[rep[at]], idx[rep[at]], thr[at])), got3[at])
    assert np.array_equal(_bit(E.evaluate_batch(v1, w[rep[at]], n[rep[at]], idx[rep[at]])), got1[rep[at]])
    for k in np.unique(np.linspace(0, len(rep) - 1, twins).astype(np.int64)):
        c, t = cases[rep[k]], int(thr[k])
        assert D.selection_ok(*c) == got1[rep[k]] and D.selection_ok_thr(c[0], c[1], c[2], t) == got3[k], (rep[k], t)
        for C in Cs:
            ok = [BD.chunk_ok(c[0][h * C:(h + 1) * C], c[1], c[2], t, T, N, C, h) for h in range(-(-N // C))]
            assert BD.final_ok(ok, c[1], c[2], T, N) == got3[k], (rep[k], t, C)
    return got1


def test_merge_sort_layers_sort_every_bit_pattern() -> None:
    """`merge_sort_layers(n)` sorts every 0-1 input up to n = 13 (so every input, by the 0-1 principle), its layers' comparators
    are disjoint, and at a power of two it is Batcher's network of (k^2 - k + 4) 2^(k-2) - 1 comparators."""
    from verity_vllm.program.registry import boolean_dsa as BD

    for n in range(1, 14):
        x = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
        for layer in BD.merge_sort_layers(n):
            ends = [e for ab in layer for e in ab]
            assert len(set(ends)) == len(ends) and all(a < b < n for a, b in layer)
            a, b = np.array(layer).T
            x[:, a], x[:, b] = np.minimum(x[:, a], x[:, b]), np.maximum(x[:, a], x[:, b])
        assert (np.diff(x, axis=1) >= 0).all(), n
    for k in range(1, 12):
        assert sum(map(len, BD.merge_sort_layers(1 << k))) == (k * k - k + 4) * (1 << k) // 4 - 1
    assert [sum(map(len, BD.merge_sort_layers(t))) for t in (2048, 1024)] == [58367, 24063]


@pytest.mark.parametrize("T, N, Cs", [(1, 2, (1, 2)), pytest.param(2, 3, (1, 2, 4), marks=pytest.mark.slow)])
def test_threshold_selection_on_every_special_row(T: int, N: int, Cs) -> None:
    """Every row of N words from `_SPECIAL` (signed zeros, infinities and NaNs of both signs, ties), every len in {INT_MIN, -1,
    0..N+1, INT_MAX} and every TOPK-tuple of indices from {-1, 0..N, 2^31}: v1 is v3 at some threshold and at
    `selection_threshold`, and v4 is v3 at every threshold class, in chunks of one position, of two with one left over, and of
    four over three positions (W < 2^w)."""
    import itertools

    lens = sorted({-(1 << 31), -1, *range(N + 2), (1 << 31) - 1})
    vals = (-1, *range(N + 1), 1 << 31)
    rows = [np.array(r, np.uint32) for r in itertools.product(_SPECIAL, repeat=N)]
    cases = [(r, n & M32, [i & M32 for i in ix]) for r in rows for n in lens for ix in itertools.product(vals, repeat=T)]
    got = _threshold_agrees(T, N, Cs, cases, ir=256)
    assert 0 < got.sum() < len(got)


def _capture_rows(T: int, N: int, seed: int) -> list[tuple[str, np.ndarray, int]]:
    """`glm53_dsa_capture_gpu._long_rows`' kinds at N keys, each (name, words, len) with +inf past len: random; ``m`` equal words
    at the threshold (`ties<m>`) or m consecutive words from 1.0 there (`crowd<m>`), m // 2 of them in the top set beside T - m // 2
    keys above and the rest below; three +0 and four -0 at the threshold; +inf among the keys; -inf at the threshold; NaNs of both
    signs."""
    rng = np.random.default_rng(seed)
    n = N - 3

    def around(middle: np.ndarray) -> np.ndarray:
        a = max(T - middle.size // 2, 0)
        v = np.concatenate([rng.uniform(100, 200, a), middle, rng.uniform(-200, -100, n - a - middle.size)]).astype(np.float32)
        return v[rng.permutation(n)]

    one = np.float32(1.0).view(np.uint32)
    rows = [("random", rng.standard_normal(n).astype(np.float32))]
    for m in sorted({2, T, 2 * T}):
        rows += [(f"ties{m}", around(np.full(m, 2.0, np.float32))),
                 (f"crowd{m}", around((one + np.arange(m, dtype=np.uint32)).view(np.float32)))]
    rows.append(("zeros", around(np.array([0.0] * 3 + [-0.0] * 4, np.float32))))
    pinf = rng.standard_normal(n).astype(np.float32)
    pinf[rng.choice(n, 3, replace=False)] = np.inf
    ninf = np.full(n, -np.inf, np.float32)
    ninf[rng.choice(n, T - max(1, T // 4), replace=False)] = rng.standard_normal(T - max(1, T // 4))
    nan = rng.standard_normal(n).astype(np.float32)
    nan[rng.choice(n, 4, replace=False)] = np.array([0x7FC00000] * 2 + [0xFFC00000] * 2, np.uint32).view(np.float32)
    rows += [("pinf", pinf), ("ninf", ninf), ("nan", nan)]
    out = []
    for name, v in rows:
        full = np.full(N, np.inf, np.float32)
        full[:n] = v
        out.append((name, full.view(np.uint32), n))
    return out


def _overflowed(words: np.ndarray, n: int, T: int) -> list[int]:
    """What a kernel past its tie capacity writes: every key above the threshold key's coarse class (its sign and exponent), then
    the class's first keys by index."""
    key = D.order_keys(words[:n])
    t = np.sort(key)[::-1][T - 1]
    coarse = (words[:n].astype(np.int64) >> 23)
    cls = coarse == coarse[np.flatnonzero(key == t)[0]]
    above = np.flatnonzero(~cls & (key > t))
    return [int(i) for i in np.concatenate([above, np.flatnonzero(cls)[:T - above.size]])]


@pytest.mark.parametrize("T, N, Cs", [pytest.param(16, 64, (16, 8), marks=pytest.mark.slow), (5, 37, (8, 4))])
def test_threshold_selection_on_the_captures_kinds_of_row(T: int, N: int, Cs) -> None:
    """The capture's kinds of long row (`_capture_rows`) at small statics, each with its exact top set in a random order, and the
    kernels' overflow (`_overflowed`), a selected +0 swapped for a -0, and the least selected key swapped for the greatest left
    out: v1 is v3 at some threshold and at `selection_threshold`, v4 (chunks of 16 or 8 over 64 keys; of 8 or 4 over 37, the last
    ragged) v3 at every threshold class, and the twins agree; the exact sets and the ties' overflow pass, the crowds' overflow and
    the swaps fail.  Then `_selection_cases`, the same way."""
    rng = np.random.default_rng(T * 1000 + N + 2)
    cases, want = [], {}
    for name, w, n in _capture_rows(T, N, T + N):
        key = D.order_keys(w[:n])
        top = _top(w, n, T, rng)
        rest = [j for j in range(n) if j not in top]
        lo, hi = min(top, key=lambda j: key[j]), max(rest, key=lambda j: key[j])
        over = _overflowed(w, n, T)
        for kind, idx, ok in (("exact", top, True), ("overflow", over, None),
                              ("swap", [hi if j == lo else j for j in top], False if key[lo] != key[hi] else None)):
            want[name, kind] = (len(cases), ok)
            cases.append((w, n, idx))
        if name == "zeros":
            plus, minus = [j for j in top if w[j] == 0], [j for j in rest if w[j] == SIGN]
            want[name, "-0 for +0"] = (len(cases), False)
            cases.append((w, n, [minus[0] if j == plus[0] else j for j in top]))
    got = _threshold_agrees(T, N, Cs, cases)
    assert {k: got[i] for k, (i, ok) in want.items() if ok is not None} == {k: ok for k, (_i, ok) in want.items() if ok is not None}
    assert got[want["ties" + str(2 * T), "overflow"][0]] == 1 and got[want["crowd" + str(2 * T), "overflow"][0]] == 0
    assert got[want["zeros", "overflow"][0]] == 1
    more, _ = _selection_cases(T, N, T * 1000 + N + 3)
    _threshold_agrees(T, N, Cs, more)


def test_threshold_twin_on_the_capture_rows() -> None:
    """The capture's own long rows (`_long_rows`, both top-ks, up to 262,144 keys): at each row's exact top set and at the kernels'
    overflow of the threshold's coarse class, v3's twin at `selection_threshold` is v1's; every exact set and every ties row's
    overflow passes, every crowd's overflow fails."""
    from tests.properties import glm53_dsa_capture_gpu as X

    for topk in X.LONG_TOPKS:
        vals, lens, names = X._long_rows(topk, 13 + topk)
        rng = np.random.default_rng(topk)
        for v, n, name in zip(vals.view(np.uint32), lens, names):
            for idx, kind in ((_top(v, n, topk, rng), "exact"), (_overflowed(v, n, topk), "overflow")):
                ok = D.selection_ok(v, n, idx)
                assert D.selection_ok_thr(v, n, idx, D.selection_threshold(v, n, idx)) == ok, (topk, name, kind)
                if kind == "exact" or name.startswith(("ties", "crowd")):
                    assert ok == (kind == "exact" or name.startswith("ties")), (topk, name, kind)


def test_v4_is_v3_on_bits_and_its_chunks_are_units_at_the_root() -> None:
    """The switch maps v3 to v4 at the served chunk (v1 still to v2); under `Q_word` and `Q_call` one Call of v4 is one unit, and
    v4's body at a Program's root (`topk_selection_calls`) is one unit per chunk and the final."""
    from verity.ml.circuits import partition_object as PO
    from verity.ml.circuits.boolean import trace as TR
    from verity.ml.circuits.defs import CompositeDefinition
    from verity.ml.circuits.program import Program
    from verity_vllm.program import boolean as SW
    from verity_vllm.program.registry import boolean_dsa as BD

    v3 = bind(D.DsaTopKSelectionV3, TOPK=16, N=64)
    assert SW.boolean_version(v3).id == bind(BD.DsaTopKSelectionV4, TOPK=16, N=64, C=BD.CHUNK).id
    assert SW.boolean_version(bind(D.DsaTopKSelection, TOPK=16, N=64)).id == bind(BD.DsaTopKSelectionV2, TOPK=16, N=64).id
    v4 = bind(BD.DsaTopKSelectionV4, TOPK=16, N=64, C=16)
    assert TR.word_of(v4).id == v3.id
    params, ret = v4.signature()
    one = Program(CompositeDefinition("OneCallOfV4", 1, (), lambda S: (tuple(params), ret), lambda B, S, *x: B.call(v4, *x),
                                      register=False))
    for q in (PO.query(), PO.call_query()):
        assert PO.evaluate(one, q).population == 1
        assert PO.evaluate(Program(v4), q).population == 5
    assert [fn.id for _k, fn in PO.calls(Program(v4))] == [f"DsaTopKChunk_v1{{TOPK=16,N=64,C=16,H={h}}}" for h in range(4)] + [
        "DsaTopKFinal_v1{TOPK=16,N=64,C=16}"]


#: v4's chunks at the served statics on the capture's rows (``VERITY_VLLM_DSA_SERVED``): GLM-5.3's top-2048 of 131,072 keys in 8
#: chunks and DeepSeek-V4-Pro's top-1024 of 262,144 in 16
SERVED = os.environ.get("VERITY_VLLM_DSA_SERVED")
_SERVED_CHUNKS = [(2048, 131072, h) for h in range(8)] + [(1024, 262144, h) for h in range(16)]


@pytest.mark.skipif(not SERVED, reason="VERITY_VLLM_DSA_SERVED: v4's 24 chunks at the served statics, each about 4.5M ANDs "
                                       "traced (a few minutes and several GB each)")
@pytest.mark.parametrize("T, N, H", _SERVED_CHUNKS)
def test_served_chunk_on_the_capture_rows(T: int, N: int, H: int) -> None:
    """Chunk H of v4 at a served statics (chunks of `CHUNK`), on bits, against `chunk_ok` on the capture's long rows at that top-k,
    each with its exact top set and the kernels' overflow, at `selection_threshold` and one key above it; on the last chunk the
    final against `final_ok` on the twins' chunk bits, and `final_ok` of those against v1's twin."""
    from tests.properties import glm53_dsa_capture_gpu as X
    from verity.ml.circuits.evaluation import bits as BS
    from verity_vllm.program.registry import boolean_dsa as BD

    C = BD.CHUNK
    vals, lens, _names = X._long_rows(T, 13 + T)
    words = vals.view(np.uint32)[:, :N]
    rng = np.random.default_rng(T)
    cases = []
    for v, n in zip(words, lens):
        m = min(int(n), N)
        for idx in (_top(v, m, T, rng), _overflowed(v, m, T)):
            t = D.selection_threshold(v, n, idx)
            cases += [(v, int(n), idx, t), (v, int(n), idx, _key_word(int(D.order_keys(np.array([t]))[0]) + 1))]
    lo, hi = H * C, min(N, (H + 1) * C)
    fn = bind(BD.DsaTopKChunk, TOPK=T, N=N, C=C, H=H)
    got = _bit(BS.evaluate_words(fn, np.stack([c[0][lo:hi] for c in cases]), np.array([c[1] for c in cases], np.uint32),
                                 np.array([c[2] for c in cases], np.uint32), np.array([c[3] for c in cases], np.uint32)))
    want = [BD.chunk_ok(c[0][lo:hi], c[1], c[2], c[3], T, N, C, H) for c in cases]
    BD.DsaTopKChunk._cache.pop(fn.key, None)
    assert got.tolist() == [int(x) for x in want]
    if hi < N:
        return
    ok = np.array([[BD.chunk_ok(c[0][h * C:(h + 1) * C], c[1], c[2], c[3], T, N, C, h) for h in range(N // C)] for c in cases])
    fin = _bit(BS.evaluate_words(bind(BD.DsaTopKFinal, TOPK=T, N=N, C=C), (ok @ (1 << np.arange(N // C))).astype(np.uint32),
                                 np.array([c[1] for c in cases], np.uint32), np.array([c[2] for c in cases], np.uint32)))
    assert fin.tolist() == [int(BD.final_ok(o, c[1], c[2], T, N)) for o, c in zip(ok, cases)]
    assert fin.tolist() == [int(D.selection_ok_thr(c[0], c[1], c[2], c[3])) for c in cases]
    assert [D.selection_ok(*c[:3]) for c in cases[::2]] == [bool(f) for f in fin[::2]]


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
    got = np.asarray(_flat(evaluate(bind(D.DsaMqaLogits, N=N, H=32, D=128), [v for r in q for v in r], w,
                                    [v for r in ks for v in r], sc)), np.uint32).view(np.float32).astype(np.float64)
    qt = torch.tensor(q, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    kt = torch.tensor(ks, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    score = torch.einsum("hd,nd->hn", qt, kt)
    ref = ((score.relu() * _t_f32(w).double()[:, None]).sum(dim=0) * _t_f32(sc).double()).numpy()
    diff = 1 - 2 * float((got * ref).sum()) / float((got * got + ref * ref).sum())
    assert diff < 1e-3
