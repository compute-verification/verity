"""DeepSeek-V4-Pro's DSA indexer kernels (`dsv4_indexer`) against models of them.

* each word Definition word for word against a model written here from the Triton kernels (vLLM d9105ea80, Triton 3.7.1's sm_90
  PTX): every add, mul and fma rounded once from the exact rational with the GPU's special values (a NaN result 0x7FFFFFFF, inf -
  inf and 0 * inf NaN), the warp butterflies simulated lane by lane, `__nv_log2f` with both of its branches and the e4m3 rounding
  from GLM-5.3's model (`tests.program.test_dsa`); MUFU's ex2, rcp and rsqrt and div.full's range scaling are core's primitives,
  each pinned by its own vectors;
* the composites (the cache entry, the head) as the compositions of their parts;
* the words the hardware makes of special values: a NaN quantises to +448 (0x7E), an infinite amax gives an infinite scale, the
  executed rotation of the nope pairs turns an infinite partner into F2FP's NaN 0x7FFF widened, the -inf scores of the rows before
  position 0 weigh nothing;
* the short-context fill against int64 `(pos + 1) // RATIO` truncating, at INT64_MAX (whose pos + 1 wraps) and negatives;
* the cache entry and the head against float64 restatements of the math within 1 fp8 step, with the same scale.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.program.registry import dsa as D
from verity_vllm.program.registry import dsv4_indexer as I
from verity_vllm.program.registry import prims as P
from tests.program import test_dsa as G

SIGN, INF, M32, NAN_PTX = G.SIGN, G.INF, G.M32, G.NAN_PTX
ABS, NEG_INF = 0x7FFFFFFF, G.NEG_INF
F_448 = D.F_448


# --- the model ---------------------------------------------------------------------------------------------------------------

def _inf(w: int) -> bool:
    return w & ABS == INF


def _zero(w: int) -> bool:
    return w & ABS == 0


def add(a: int, b: int) -> int:
    """add.f32 (round to nearest, no flush): a NaN operand or inf - inf gives 0x7FFFFFFF."""
    if G._is_nan(a) or G._is_nan(b) or (_inf(a) and _inf(b) and a != b):
        return NAN_PTX
    return a if _inf(a) else b if _inf(b) else G._fadd(a, b)


def sub(a: int, b: int) -> int:
    return add(a, b ^ SIGN)


def mul(a: int, b: int) -> int:
    if G._is_nan(a) or G._is_nan(b) or (_inf(a) and _zero(b)) or (_zero(a) and _inf(b)):
        return NAN_PTX
    if _inf(a) or _inf(b):
        return INF | ((a ^ b) & SIGN)
    return G._fmul(a, b)


def fma(a: int, b: int, c: int) -> int:
    """fma.rn.f32: the exact a * b + c rounded once; a NaN operand, 0 * inf, or an infinite product against the opposite infinity
    gives 0x7FFFFFFF."""
    if G._is_nan(c):
        return NAN_PTX
    if G._is_nan(a) or G._is_nan(b) or _inf(a) or _inf(b):
        p = mul(a, b)
        return p if G._is_nan(p) else add(p, c)
    return c if _inf(c) else G._ffma(a, b, c)


def fma_ftz(a: int, b: int, c: int) -> int:
    return G._ftz(fma(G._ftz(a), G._ftz(b), G._ftz(c)))


def ex2(x: int) -> int:
    return P.MufuEx2Ftz.evaluate(x)


def div_full(a: int, b: int) -> int:
    """div.full.f32 as ptxas emits it: the range-scaled numerator times MUFU.RCP of the range-scaled divisor."""
    return mul(P.DivFullScaleA.evaluate(a, b), P.DivFullRcp.evaluate(b))


def ex2_approx(t: int) -> int:
    """ex2.approx.f32 (no .ftz): below -126 (ordered) MUFU.EX2 of t / 2, squared."""
    if not G._is_nan(t) and G._val(t) < -126.0:
        y = ex2(mul(t, 0x3F000000))
        return mul(y, y)
    return ex2(t)


def fmin(a: int, b: int) -> int:
    """min.f32: a NaN operand yields the other (two give 0x7FFFFFFF), min(+0, -0) = -0."""
    if G._is_nan(a):
        return NAN_PTX if G._is_nan(b) else b
    if G._is_nan(b):
        return a
    if G._val(a) == G._val(b):
        return a | b
    return a if G._val(a) < G._val(b) else b


def clamp_448(x: int) -> int:
    """min.xorsign.abs.f32(x, 448)."""
    return fmin(x & ABS, F_448) | (x & SIGN)


def exponent(amax: int, eps: int) -> int:
    """cvt.rpi.ftz of `__nv_log2f(max(amax, eps) * f32(1/448))`."""
    return G._ceil_ftz(G.log2f(mul(G._fmax(amax, eps), I.INV_448)))


def bf16(w: int) -> int:
    """cvt.rn.bf16.f32 (F2FP): a NaN gives 0x7FFF."""
    return 0x7FFF if G._is_nan(w) else ((w + 0x7FFF + ((w >> 16) & 1)) >> 16) & 0xFFFF


def column(kv: list[int], score: list[int]) -> int:
    """fused_compress_quant_cache.py:769-777 for one column."""
    m = score[0]
    for x in score[1:]:
        m = G._fmax(m, x)
    e = [ex2_approx(mul(sub(x, m), I.LOG2E)) for x in score]
    den = e[0]
    for x in e[1:]:
        den = add(den, x)
    s = [div_full(x, den) for x in e]
    if len(kv) == 1:
        return mul(s[0], kv[0])
    c = fma(s[0], kv[0], mul(s[1], kv[1]))
    for r in range(2, len(kv)):
        c = fma(s[r], kv[r], c)
    return c


def norm_rope(c: list[int], w: list[int], cs: list[int], eps: float) -> list[int]:
    """fused_compress_quant_cache.py:781-831 on one warp: lane l holds columns l + 32k, its squares summed from column 32 + l's
    (`mul`, then `fma` of k = 0, 2, 3), the xor butterfly of add.f32 over the lanes; the RMSNorm, the rotation of every pair (the
    nope pairs by 1.0 and 0.0), the bf16 rounding."""
    Dd, rope = len(c), len(cs)
    order = (1, 0, 2, 3) if Dd == 128 else (0,)
    lanes = []
    for lane in range(32):
        cols = [c[lane + 32 * k] for k in order]
        acc = mul(cols[0], cols[0])
        for x in cols[1:]:
            acc = fma(x, x, acc)
        lanes.append(acc)
    total = G._butterfly(lanes, add, (16, 8, 4, 2, 1))
    rrms = P.RsqrtApprox.evaluate(add(D.f32_bits(eps), div_full(total, D.f32_bits(Dd))))
    n = [mul(mul(c[i], rrms), w[i] << 16) for i in range(Dd)]
    nope, half = (Dd - rope) // 2, rope // 2
    out = []
    for j in range(Dd // 2):
        cos, sin = (I.ONE, 0) if j < nope else (cs[j - nope], cs[half + j - nope])
        e, o = n[2 * j], n[2 * j + 1]
        out += [fma(e, cos, mul(o, sin) ^ SIGN), fma(o, cos, mul(e, sin))]
    return [bf16(x) << 16 for x in out]


def _amax(x: list[int]) -> int:
    """max.f32 over |x| (in any order: the max of words with their sign cleared drops NaNs and is associative)."""
    a = x[0] & ABS
    for v in x[1:]:
        a = G._fmax(a, v & ABS)
    return a


def k_quant(x: list[int]) -> list[int]:
    """fused_compress_quant_cache.py:832-847: the bytes, then the scale."""
    e = exponent(_amax(x), I.K_EPS)
    inv = ex2(sub(0, e))
    return [G._e4m3(clamp_448(mul(inv, v))) for v in x] + [ex2(e)]


def compress(kv: list[list[int]], score: list[list[int]], w: list[int], cs: list[int], eps: float) -> list[int]:
    Dd = len(w)
    c = [column([kv[r][j] for r in range(8)], [score[r][j] for r in range(8)]) for j in range(Dd)]
    return k_quant(norm_rope(c, w, cs, eps))


def inv_rope(x: list[int], cs: list[int]) -> list[int]:
    """fused_inv_rope_fp8_quant.py:89-106: the even word `fma(x, cos_j, x' * sin_j)`, the odd one `fma(x, cos_j, -(x' * sin_j))`."""
    R = len(x)
    xf = [v << 16 for v in x]
    out = []
    for i in range(R):
        t = mul(xf[i ^ 1], cs[R // 2 + i // 2])
        out.append(fma(xf[i], cs[i // 2], t if i % 2 == 0 else t ^ SIGN))
    return out


def inv_quant(x: list[int]) -> list[int]:
    """fused_inv_rope_fp8_quant.py:110-121: the bytes, then the scale."""
    scale = ex2(exponent(_amax(x), I.INV_ROPE_EPS))
    return [G._e4m3(clamp_448(div_full(v, scale))) for v in x] + [scale]


def head(x: list[int], cs: list[int], qg: int) -> list[int]:
    rope = len(cs)
    words = [v << 16 for v in x[:len(x) - rope]] + inv_rope(x[len(x) - rope:], cs)
    parts = [inv_quant(words[g:g + qg]) for g in range(0, len(words), qg)]
    return [b for p in parts for b in p[:-1]] + [p[-1] for p in parts]


def fill(pos: int, topk: int, ratio: int) -> list[int]:
    """attention.py:71-88: `(pos + 1) // ratio` in int64 (truncating, pos + 1 wrapping), slot i valid below it."""
    q = (pos + 1 + (1 << 63)) % (1 << 64) - (1 << 63)
    n = abs(q) // ratio * (1 if q >= 0 else -1)
    return [i if i < n else M32 for i in range(topk)]


# --- inputs ------------------------------------------------------------------------------------------------------------------

def _f32(rng, n: int, std: float = 1.0, mean: float = 0.0) -> list[int]:
    return G._f32(rng, n, mean, std)


def _cs(rng, rope: int) -> list[int]:
    """The f32 rotary cache row at one position: [cos | sin]."""
    half = rope // 2
    theta = int(rng.integers(0, 1 << 16)) / 10000.0 ** (np.arange(half) / half)
    return np.concatenate([np.cos(theta), np.sin(theta)]).astype(np.float32).view(np.uint32).astype(int).tolist()


def _state(rng, Dd: int, std: float = 1.0) -> tuple[list[list[int]], list[list[int]]]:
    return [_f32(rng, Dd, std) for _ in range(8)], [_f32(rng, Dd, 2.0) for _ in range(8)]


def _flat(out) -> list[int]:
    return [int(v) for v in out]


def _rows(a: list[list[int]]) -> list[int]:
    return [v for r in a for v in r]


def _signed(words: list[int]) -> list[int]:
    return [w - (1 << 32) if w >> 31 else w for w in words]


COMPRESS = bind(I.Dsv4IndexerCompress, D=128, ROPE=64, EPS=1e-6)
HEAD = bind(I.Dsv4InvRopeQuantHead, CHUNKS=4, QG=128, ROPE=64)


# --- step 1, the score -------------------------------------------------------------------------------------------------------

def test_score_is_the_sum_with_the_ape_row() -> None:
    rng = np.random.default_rng(1)
    fn = bind(I.Dsv4CompressorScore, W=8)
    score, ape = _f32(rng, 8), _f32(rng, 8, 0.1)
    score[:4] = [NAN_PTX, INF, NEG_INF, 0x7F7FFFFF]
    ape[:4] = [0, NEG_INF, NEG_INF, 0x7F7FFFFF]
    want = [add(s, a) for s, a in zip(score, ape)]
    assert _flat(evaluate(fn, score, ape)) == want
    assert want[:4] == [NAN_PTX, NAN_PTX, NEG_INF, INF]


# --- step 2, the compressor --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rows", [1, 2, 8])
def test_column_matches_the_kernel_model(rows: int) -> None:
    rng = np.random.default_rng(rows)
    fn = bind(I.Dsv4CompressColumn, ROWS=rows)
    for std in (1e-3, 1.0, 1.0, 30.0, 300.0):
        kv, score = _f32(rng, rows), _f32(rng, rows, std)
        assert evaluate(fn, kv, score)[0] == column(kv, score)


def test_column_edge_words() -> None:
    """The rows before position 0 (score -inf, kv 0) weigh exactly nothing; a NaN score or kv makes the column NaN; scores far
    apart send ex2.approx below -126, where ptxas halves and squares."""
    fn = bind(I.Dsv4CompressColumn, ROWS=8)
    rng = np.random.default_rng(9)
    kv, score = _f32(rng, 8), _f32(rng, 8)
    cases = [([0] * 4 + kv[4:], [NEG_INF] * 4 + score[4:]),
             (kv, [NAN_PTX] + score[1:]),
             ([INF] + kv[1:], score),
             (kv, [D.f32_bits(v) for v in (0.0, -100.0, -95.0, -88.5, -87.0, -1.0, -120.0, -0.5)])]
    for k, s in cases:
        assert evaluate(fn, k, s)[0] == column(k, s)
    assert evaluate(fn, *cases[1])[0] == NAN_PTX
    t = mul(D.f32_bits(-100.0), I.LOG2E)                                                  # 2^-144: subnormal, not flushed
    assert ex2_approx(t) != 0 and ex2(t) == 0


@pytest.mark.parametrize("Dd,rope", [(32, 16), (128, 64)])
def test_norm_rope_matches_the_kernel_model(Dd: int, rope: int) -> None:
    rng = np.random.default_rng(Dd)
    fn = bind(I.Dsv4IndexerNormRope, D=Dd, ROPE=rope, EPS=1e-6)
    for std in (1e-3, 1.0, 1.0, 50.0):
        c, w, cs = _f32(rng, Dd, std), G._f32(rng, Dd, 1.0, 0.2), _cs(rng, rope)
        w = [v >> 16 for v in w]
        assert _flat(evaluate(fn, c, w, cs)) == norm_rope(c, w, cs, 1e-6)
    c = [0] * Dd                                                                          # the sum 0: rrms = rsqrt(eps)
    assert _flat(evaluate(fn, c, w, cs)) == norm_rope(c, w, cs, 1e-6)


def test_the_nope_pairs_are_rotated_by_one_and_zero() -> None:
    """The kernel rotates the first (D - ROPE)/2 pairs by cos 1.0 and sin 0.0 too, so an infinite normed word makes its partner NaN
    (inf * 0): F2FP's 0x7FFF widened, where skipping the rotation would keep the partner."""
    rng = np.random.default_rng(11)
    fn = bind(I.Dsv4IndexerNormRope, D=32, ROPE=16, EPS=1e-6)
    c, cs = _f32(rng, 32), _cs(rng, 16)
    w = [0x3F80] * 32
    w[3] = 0x7F80                                                                         # n[3] = +inf, in the nope pair (2, 3)
    got = _flat(evaluate(fn, c, w, cs))
    assert got == norm_rope(c, w, cs, 1e-6)
    assert got[2] == 0x7FFF0000 and got[3] == (INF if c[3] >> 31 == 0 else NEG_INF)
    c = [INF] + _f32(rng, 31)                                                             # the sum inf: rrms 0, n[0] = inf * 0
    got = _flat(evaluate(fn, c, [0x3F80] * 32, cs))
    assert got == norm_rope(c, [0x3F80] * 32, cs, 1e-6) and got[:2] == [0x7FFF0000] * 2
    assert all(v & ABS == 0 for v in got[2:])


@pytest.mark.parametrize("N", [4, 32, 128])
def test_k_quant_matches_the_kernel_model(N: int) -> None:
    rng = np.random.default_rng(N)
    fn = bind(I.Dsv4IndexerKQuant, N=N)
    rows = [_f32(rng, N, s) for s in (1e-6, 1e-3, 0.3, 7.0, 300.0, 1e5)]
    rows += [[0] * N, [SIGN] * N, [0x00000001] * N, [D.f32_bits(448.0 * 2.0 ** -3) + d for d in range(N)]]
    for x in rows:
        assert _flat(evaluate(fn, x)) == k_quant(x)


def test_k_quant_edge_words() -> None:
    """A NaN quantises to +448 (0x7E); an infinite amax gives the scale +inf and inv 0, so the finite words quantise to signed zeros
    and the infinite one to 0 * inf = NaN, +448; an amax below 1e-4 takes the floor's scale 2^-22."""
    fn = bind(I.Dsv4IndexerKQuant, N=4)
    one, mone = 0x3F800000, 0xBF800000
    for x, bytes_, scale in [([NAN_PTX, one, mone, 0], None, None),
                             ([INF, one, mone, SIGN], [0x7E, 0x00, 0x80, 0x80], INF),
                             ([NEG_INF, one, 0, 0], [0x7E, 0x00, 0x00, 0x00], INF),
                             ([D.f32_bits(1e-6)] * 4, None, D.f32_bits(2.0 ** -22))]:
        got = _flat(evaluate(fn, x))
        assert got == k_quant(x), [hex(v) for v in x]
        if bytes_ is not None:
            assert got[:4] == bytes_
        if scale is not None:
            assert got[4] == scale
    assert _flat(evaluate(fn, [NAN_PTX, one, mone, 0]))[0] == 0x7E


def test_compress_matches_the_kernel_model() -> None:
    rng = np.random.default_rng(13)
    w = [v >> 16 for v in G._f32(rng, 128, 1.0, 0.2)]
    for std in (0.01, 1.0, 40.0):
        kv, score = _state(rng, 128, std)
        cs = _cs(rng, 64)
        assert _flat(evaluate(COMPRESS, _rows(kv), _rows(score), w, cs)) == compress(kv, score, w, cs, 1e-6)
    kv, score = _state(rng, 128)                                                          # position 3: rows 0-3 before 0
    kv[:4] = [[0] * 128 for _ in range(4)]
    score[:4] = [[NEG_INF] * 128 for _ in range(4)]
    assert _flat(evaluate(COMPRESS, _rows(kv), _rows(score), w, cs)) == compress(kv, score, w, cs, 1e-6)


def test_compress_is_its_parts() -> None:
    rng = np.random.default_rng(14)
    kv, score = _state(rng, 32)
    w, cs = [v >> 16 for v in G._f32(rng, 32, 1.0, 0.2)], _cs(rng, 16)
    col = bind(I.Dsv4CompressColumn, ROWS=8)
    c = [evaluate(col, [kv[r][j] for r in range(8)], [score[r][j] for r in range(8)])[0] for j in range(32)]
    v = _flat(evaluate(bind(I.Dsv4IndexerNormRope, D=32, ROPE=16, EPS=1e-6), c, w, cs))
    want = _flat(evaluate(bind(I.Dsv4IndexerKQuant, N=32), v))
    assert _flat(evaluate(bind(I.Dsv4IndexerCompress, D=32, ROPE=16, EPS=1e-6), _rows(kv), _rows(score), w, cs)) == want


def test_the_statics_are_checked() -> None:
    with pytest.raises(ValueError):
        evaluate(bind(I.Dsv4IndexerNormRope, D=64, ROPE=16, EPS=1e-6), [0] * 64, [0] * 64, [0] * 16)
    with pytest.raises(ValueError):
        evaluate(bind(I.Dsv4IndexerKQuant, N=6), [0] * 6)
    with pytest.raises(ValueError):
        evaluate(bind(I.Dsv4InvRope, ROPE=3), [0] * 3, [0] * 3)
    with pytest.raises(ValueError):
        evaluate(bind(I.Dsv4InvRopeQuantHead, CHUNKS=2, QG=4, ROPE=8), [0] * 8, [0] * 8)
    with pytest.raises(ValueError):
        evaluate(bind(I.Dsv4ShortContextTopK, TOPK=1 << 30, RATIO=4), 0)


# --- step 3, the inverse RoPE and its quantisation ---------------------------------------------------------------------------

def test_inv_rope_matches_the_kernel_model() -> None:
    rng = np.random.default_rng(15)
    fn = bind(I.Dsv4InvRope, ROPE=64)
    for scale in (0.01, 1.0, 1e4):
        x, cs = G._bf16(rng, 64, scale), _cs(rng, 64)
        assert _flat(evaluate(fn, x, cs)) == inv_rope(x, cs)
    x[0], x[3], x[5] = 0x7F80, 0xFFC0, 0x0001
    assert _flat(evaluate(fn, x, cs)) == inv_rope(x, cs)


@pytest.mark.parametrize("N", [4, 128])
def test_inv_rope_quant_matches_the_kernel_model(N: int) -> None:
    rng = np.random.default_rng(16 + N)
    fn = bind(I.Dsv4InvRopeQuant, N=N)
    rows = [_f32(rng, N, s) for s in (1e-12, 1e-6, 1.0, 1e3, 1e30)]
    rows += [[0] * N, [0x00000001] * N, [D.f32_bits(448.0 * 2.0 ** 5) + d for d in range(N)],
             [NAN_PTX] + _f32(rng, N - 1), [INF] + _f32(rng, N - 1), [0x7F7FFFFF] + _f32(rng, N - 1)]
    for x in rows:
        assert _flat(evaluate(fn, x)) == inv_quant(x)


def test_inv_rope_quant_edge_words() -> None:
    """An all-zero group takes the floor's scale 2^-42 (1e-10 / 448 rounds up to it); NaN gives +448; an infinite amax gives the
    scale +inf, and div.full by it a NaN for the infinite word (+448) and signed zeros for the rest."""
    fn = bind(I.Dsv4InvRopeQuant, N=4)
    one, mone = 0x3F800000, 0xBF800000
    assert _flat(evaluate(fn, [0, SIGN, 0, 0])) == [0, 0x80, 0, 0, D.f32_bits(2.0 ** -42)]
    assert _flat(evaluate(fn, [NAN_PTX, one, mone, 0]))[0] == 0x7E
    assert _flat(evaluate(fn, [INF, one, mone, 0])) == [0x7E, 0x00, 0x80, 0x00, INF]


def test_div_full_by_a_power_of_two_is_the_exact_quotient() -> None:
    """The scales are powers of two, so the quantised words are the exactly rounded quotients (as GLM's `_div_by_scale` has them)
    while the scale stays at most 2^126; above it div.full's range scaling still keeps them exact."""
    rng = np.random.default_rng(17)
    xs = _f32(rng, 60, 1.0) + _f32(rng, 60, 1e30) + _f32(rng, 60, 1e-30) + [0x00000001, 0x007FFFFF, 0x7F7FFFFF, SIGN]
    for k in range(-42, 128):
        s = D.f32_bits(2.0 ** k)
        r = P.DivFullRcp.evaluate(s)
        for x in xs:
            assert mul(P.DivFullScaleA.evaluate(x, s), r) == G._div_by_scale(x, s), (hex(x), k)


def test_head_matches_the_kernel_model() -> None:
    rng = np.random.default_rng(18)
    for scale in (0.01, 1.0, 300.0):
        x, cs = G._bf16(rng, 512, scale), _cs(rng, 64)
        assert _flat(evaluate(HEAD, x, cs)) == head(x, cs, 128)
    x[7], x[500] = 0x7FC0, 0x7F80
    assert _flat(evaluate(HEAD, x, cs)) == head(x, cs, 128)


def test_head_is_its_parts() -> None:
    rng = np.random.default_rng(19)
    x, cs = G._bf16(rng, 16), _cs(rng, 4)
    rot = _flat(evaluate(bind(I.Dsv4InvRope, ROPE=4), x[12:], cs))
    words = [v << 16 for v in x[:12]] + rot
    q = bind(I.Dsv4InvRopeQuant, N=8)
    a, b = _flat(evaluate(q, words[:8])), _flat(evaluate(q, words[8:]))
    got = _flat(evaluate(bind(I.Dsv4InvRopeQuantHead, CHUNKS=2, QG=8, ROPE=4), x, cs))
    assert got == a[:8] + b[:8] + [a[8], b[8]]


# --- step 4, the short-context fill ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("topk,ratio", [(16, 4), (1024, 4), (16, 128)])
def test_fill_is_the_truncating_int64_quotient(topk: int, ratio: int) -> None:
    fn = bind(I.Dsv4ShortContextTopK, TOPK=topk, RATIO=ratio)
    big = (1 << 63) - 1
    lim = ratio * topk
    positions = [-(1 << 63), -(1 << 63) + 1, -9, -5, -4, -2, -1, 0, 1, ratio - 2, ratio - 1, ratio, 2 * ratio - 1, lim - 2, lim - 1,
                 lim, lim + 1, (1 << 31) - 1, 1 << 31, (1 << 32) + 3, 1 << 40, big - ratio, big - 1, big]
    rng = np.random.default_rng(topk + ratio)
    positions += [int(p) for p in rng.integers(0, lim + 8, 6)]
    for p in positions:
        assert _signed(_flat(evaluate(fn, p & ((1 << 64) - 1)))) == _signed(fill(p, topk, ratio)), p


# --- float64 restatements ----------------------------------------------------------------------------------------------------

def _v(words) -> np.ndarray:
    return np.asarray(words, np.uint32).view(np.float32).astype(np.float64)


def _ref_quant(v: np.ndarray, eps: float) -> tuple[list[int], int]:
    """ceil(log2(max(amax, eps) / 448)) and the bytes of v / 2^that, each rounded once from float64."""
    e = math.ceil(math.log2(max(float(np.abs(v).max()), eps) / 448.0))
    return [G._e4m3(D.f32_bits(x / 2.0 ** e)) for x in v], D.f32_bits(2.0 ** e)


def _ref_rotate(x: np.ndarray, cs: np.ndarray, inverse: bool) -> np.ndarray:
    half = len(cs) // 2
    c, s = cs[:half], (-1 if inverse else 1) * cs[half:]
    out = x.copy()
    e, o = x[0::2].copy(), x[1::2].copy()
    out[0::2], out[1::2] = e * c - o * s, o * c + e * s
    return out


def test_compress_within_a_float64_reference() -> None:
    """The softmax over the 8 rows, the weighted sum, the RMSNorm, the rotation of the last 64 in float64, rounded to bf16 as the
    kernel stores them (fused_compress_quant_cache.py:831), and the ue8m0 quantisation: each byte within 1 fp8 step (the kernel's
    ex2.approx, div.full and rsqrt.approx move it at most that far) and the same scale."""
    rng = np.random.default_rng(20)
    w = [v >> 16 for v in G._f32(rng, 128, 1.0, 0.2)]
    for std in (0.01, 1.0, 40.0):
        kv, score = _state(rng, 128, std)
        cs = _cs(rng, 64)
        got = _flat(evaluate(COMPRESS, _rows(kv), _rows(score), w, cs))
        k, s = np.stack([_v(r) for r in kv]), np.stack([_v(r) for r in score])
        p = np.exp(s - s.max(axis=0))
        c = (p / p.sum(axis=0) * k).sum(axis=0)
        n = c / np.sqrt((c * c).mean() + 1e-6) * _v([v << 16 for v in w])
        n[64:] = _ref_rotate(n[64:], _v(cs), inverse=False)
        n = _v([bf16(D.f32_bits(x)) << 16 for x in n])
        qb, scale = _ref_quant(n, 1e-4)
        assert G._fp8_steps(got[:128], qb) <= 1 and got[128] == scale


def test_head_within_a_float64_reference() -> None:
    rng = np.random.default_rng(21)
    for scale in (0.01, 1.0, 300.0):
        x, cs = G._bf16(rng, 512, scale), _cs(rng, 64)
        got = _flat(evaluate(HEAD, x, cs))
        v = _v([b << 16 for b in x])
        v[448:] = _ref_rotate(v[448:], _v(cs), inverse=True)
        for g in range(4):
            qb, sc = _ref_quant(v[128 * g:128 * (g + 1)], 1e-10)
            assert G._fp8_steps(got[128 * g:128 * (g + 1)], qb) <= 1 and got[512 + g] == sc
