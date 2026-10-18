"""The sm_90 block-FP8 linear (`deepgemm`, vLLM d9105ea80's `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`) and DeepSeek-V4's
grouped `wo_a` einsum, against models of their kernels and against their twins (`kernels.deepgemm_twins`):

* the ue8m0 scale's bit form against `exp2f(ceilf(log2f(fmaxf(fabsf(y), 1e-10f))))` with `__nv_log2f` as `test_dsa` models it, and
  against the exact power of two, on every scale the kernel reaches (every bf16 group amax, the eps, +inf), with the margin that
  keeps log2f's error from carrying the ceiling;
* FlashInfer's `scale_1x128_kernel` simulated lane by lane (`shfl.down` with out-of-range lanes keeping their own value,
  `std::max` as `setp.lt.ftz` + `selp`), at the guards its FTZ compare and `setp.neu` reach: all-zero, all-subnormal (lane 0 zero
  or not), infinities and NaNs;
* each word Definition word for word against its twin on random rows and on rows that reach the guards, the linears and the
  grouped einsum over a partial weight block;
* the replay's evaluator of the three linears, their row kernels (`kernels.deepgemm_rows`), and the rows they decline;
* the Boolean versions (`boolean_deepgemm`) against their word views, the switch mapping each word Definition to them;
* the load-time weight processing: the e8m0 upcast, and the ue8m0 requantisation against vLLM's own `per_block_cast_to_fp8`
  run eagerly on CPU, off the blocks whose power of two the GPU's division and log2 decide (`requant_band`).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.program import boolean as SW
from verity_vllm.program.kernels import deepgemm_twins as T
from verity_vllm.program.kernels import derived_rows as R
from verity_vllm.program.registry import boolean_deepgemm as BDG
from verity_vllm.program.registry import deepgemm as DG
from verity_vllm.program.registry import prims as P
from tests.program.test_boolean_fp8 import _flat, _inputs
from tests.program.test_dsa import _ceil_ftz, _e4m3, _ex2_integer, _ftz, _is_nan, _val, log2f
from tests.program.test_fp8_moe import _q, _rn

SIGN, INF = 1 << 31, 0x7F800000
F448, ONE = 0x43E00000, 0x3F800000


def _bf16_rows(rng, n: int, K: int, scale: float = 1.0) -> np.ndarray:
    return ((rng.standard_normal((n, K)).astype(np.float32) * np.float32(scale)).view(np.uint32) >> 16).astype(np.uint16)


def _guard_rows(rng, K: int) -> list[np.ndarray]:
    """Rows of K bf16 words, each 128-word group reaching one guard of the quantisers."""
    sub = lambda: (rng.integers(1, 0x80, 128) | (rng.integers(0, 2, 128) << 15)).astype(np.uint16)   # nonzero subnormals
    groups = []
    g = sub(); groups.append(g)                                        # all subnormal, lane 0 nonzero: scale +inf, words 0x7F
    g = sub(); g[[0, 1, 64, 65]] = 0; groups.append(g)                 # lane 0 zero, the rest subnormal: amax 0, scale 1
    groups.append(np.zeros(128, np.uint16))                            # all +0
    groups.append(np.full(128, 0x8000, np.uint16))                     # all -0
    g = _bf16_rows(rng, 1, 128)[0]; g[7] = 0x7F80; groups.append(g)    # +inf: amax inf, stored scale inf
    g = _bf16_rows(rng, 1, 128)[0]; g[70] = 0x7FC1; groups.append(g)   # NaN: dropped from the amax, its word 0x7F
    g = _bf16_rows(rng, 1, 128)[0]; g[3] = 0xFF80; g[90] = 0xFFC0; groups.append(g)
    g = _bf16_rows(rng, 1, 128, 1e-38)[0]; groups.append(g)            # normals and subnormals mixed
    g = _bf16_rows(rng, 1, 128, 3e38)[0]; groups.append(g)             # near the largest finite, some +-inf
    g = np.full(128, 0x0080, np.uint16); g[5] = 0x0081; groups.append(g)   # the smallest normals
    g = _bf16_rows(rng, 1, 128)[0]; g[2] = 0x43E0; groups.append(g)    # 448: amax / 448 = 1
    flat = np.concatenate(groups)
    return [np.resize(np.roll(flat, -128 * i), K) for i in range(len(groups))]


def _words(x) -> list[int]:
    return [int(v) for v in np.asarray(x).reshape(-1)]


# --- the ue8m0 scale -------------------------------------------------------------------------------------------------------------

def _reachable_scales() -> np.ndarray:
    """Every `y` the generic kernel's ue8m0 path reaches: `div.rn(m, 448)` for m the eps (1e-10f) and every positive bf16 at or above
    it (an amax is an `fmaxf` fold from the eps over |x|), with +inf."""
    a = (np.arange(0x0001, 0x7F81, dtype=np.uint32) << 16)
    m = np.concatenate([np.array([DG.F_1EM10], np.uint32), a[R.as_f32(a) >= np.float32(1e-10)]])
    return R.div_rn(m, np.full(m.size, F448, np.uint32))


def _exact_pow2_ceil(w: int) -> int:
    m = max(_q(w), _q(DG.F_1EM10))
    e = m.numerator.bit_length() - m.denominator.bit_length()
    e += Fraction(2) ** e < m
    e -= Fraction(2) ** (e - 1) >= m
    return (e + 127) << 23


def test_ue8m0_bit_form_is_the_kernels_exp2f_ceilf_log2f_on_every_reachable_scale() -> None:
    ys = _reachable_scales()
    got = T.ue8m0_ceil(ys)
    for y, g in zip(_words(ys), _words(got)):
        if y == INF:
            assert g == INF
            continue
        m = y if _val(y) >= _val(DG.F_1EM10) else DG.F_1EM10            # fmaxf(fabsf(y), 1e-10f)
        assert g == _ex2_integer(_ceil_ftz(log2f(m))) == _exact_pow2_ceil(y), hex(y)


def test_ue8m0_reachable_scales_keep_clear_of_the_powers_of_two() -> None:
    """A reachable scale is a power of two or at least ~1/224 relative from one, far beyond log2f's ulp (< 2^-16 here)."""
    ys = R.as_f32(_reachable_scales()).astype(np.float64)
    ys = ys[np.isfinite(ys)]
    lg = np.log2(np.maximum(ys, 1e-10))
    frac = np.ceil(lg) - lg
    exact = frac == 0
    assert exact.any() and (frac[~exact] > 1e-3).all() and (1 - frac[~exact] > 1e-3).all()


def test_ue8m0_ceil_definition_matches_its_twin_on_edge_words() -> None:
    rng = np.random.default_rng(1)
    ys = np.concatenate([rng.integers(0, 1 << 32, 300, dtype=np.uint64).astype(np.uint32),
                         np.array([0, SIGN, INF, SIGN | INF, 0x7FC00000, 0xFFFFFFFF, 1, DG.F_1EM10, DG.F_1EM10 - 1, DG.F_1EM10 + 1,
                                   ONE, ONE + 1, ONE - 1, 0x7F7FFFFF, 0x7F000000, 0x00800000], np.uint32)])
    assert [evaluate(DG.Fp8Ue8m0Ceil, y)[0] for y in _words(ys)] == _words(T.ue8m0_ceil(ys))


# --- FlashInfer's scale_1x128_kernel, lane by lane ---------------------------------------------------------------------------------

def _lt_ftz(a: int, b: int) -> bool:
    return not (_is_nan(a) or _is_nan(b)) and _val(_ftz(a)) < _val(_ftz(b))


def _max_bf16(a: int, b: int) -> int:
    """max.bf16 of two non-negative bf16-widened words: exact, a NaN operand dropped."""
    if _is_nan(b):
        return a
    return b if _val(b) > _val(a) else a


def _mul_ftz(a: int, b: int) -> int:
    """mul.ftz.f32 (a NaN result's payload is immaterial here: it only reaches the e4m3 cast)."""
    a, b = _ftz(a), _ftz(b)
    mag = {a & 0x7FFFFFFF, b & 0x7FFFFFFF}
    if _is_nan(a) or _is_nan(b) or mag == {0, INF}:
        return 0x7FFFFFFF
    if INF in mag:
        return ((a ^ b) & SIGN) | INF
    return _ftz(_rn(_q(a) * _q(b), (a ^ b) & SIGN))


def _rcp(x: int) -> int:
    """`MufuRcpFtz_v1` (core's MUFU.RCP table)."""
    return evaluate(P.MufuRcpFtz, x)[0]


def _fi_group_model(x: list[int]) -> tuple[list[int], int]:
    """One group as the warp runs it: lane l's four elements, the shfl.down tree (lane t reads t + o, a lane past 31 its own), lane 0's
    amax, then the scale, the words and the stored scale."""
    f = [w << 16 for w in x]
    v = []
    for l in range(32):
        acc = 0
        for i in (2 * l, 2 * l + 1, 64 + 2 * l, 65 + 2 * l):
            acc = _max_bf16(acc, f[i] & 0x7FFFFFFF)
        v.append(acc)
    for o in (16, 8, 4, 2, 1):
        other = [v[t + o] if t + o < 32 else v[t] for t in range(32)]
        v = [other[t] if _lt_ftz(v[t], other[t]) else v[t] for t in range(32)]
    amax = v[0]
    nz = amax != 0
    scale = _mul_ftz(_rcp(amax), F448) if nz else ONE
    return [_e4m3(_mul_ftz(scale, w)) for w in f], _rcp(scale)


def test_flashinfer_group_against_the_warp_model() -> None:
    rng = np.random.default_rng(2)
    rows = _guard_rows(rng, 128) + list(_bf16_rows(rng, 6, 128, 4.0))
    for x in rows:
        q, s = _fi_group_model(_words(x))
        got = evaluate(DG.FlashInferFp8Quant1x128Group, _words(x))
        assert list(got[:128]) == q and got[128] == s, [hex(w) for w in x[:8]]


def test_flashinfer_guards() -> None:
    rng = np.random.default_rng(3)
    sub, lane0_zero, zeros = _guard_rows(rng, 128)[:3]
    q, s = T.flashinfer_quant_1x128_row(sub)
    assert int(s[0]) == 0 and set(_words(q)) == {0x7F}                 # amax subnormal: scale +inf, every word inf * 0
    q, s = T.flashinfer_quant_1x128_row(lane0_zero)
    assert int(s[0]) == ONE and set(_words(q) ) <= {0x00, 0x80}        # every compare false: lane 0's +0
    q, s = T.flashinfer_quant_1x128_row(zeros)
    assert int(s[0]) == ONE and set(_words(q)) == {0}


def test_f32_lt_ftz_against_its_model() -> None:
    rng = np.random.default_rng(4)
    edges = [0, SIGN, 1, SIGN | 1, 0x007FFFFF, 0x00800000, SIGN | 0x00800000, ONE, ONE + 1, INF, SIGN | INF, 0x7FC00000, 0xFFC00001]
    ws = edges + _words(rng.integers(0, 1 << 32, 40, dtype=np.uint64))
    for a in ws:
        for b in ws:
            assert evaluate(DG.F32LtFtz, a, b)[0] == int(_lt_ftz(a, b)), (hex(a), hex(b))


# --- the word Definitions against their twins -----------------------------------------------------------------------------------

@pytest.mark.parametrize("K", [128, 384])
def test_quantisers_match_their_twins(K: int) -> None:
    rng = np.random.default_rng(K)
    gq, fq = bind(DG.Fp8GroupQuantUe8m0, K=K, G=128), bind(DG.FlashInferFp8Quant1x128, K=K)
    rows = _guard_rows(rng, K) + list(_bf16_rows(rng, 4, K, 3.0)) + [rng.integers(0, 1 << 16, K).astype(np.uint16)]
    for x in rows:
        q, s = T.ue8m0_group_quant_row(x)
        assert list(evaluate(gq, _words(x))) == _words(q) + _words(s)
        q, s = T.flashinfer_quant_1x128_row(x)
        assert list(evaluate(fq, _words(x))) == _words(q) + _words(s)


def test_group_quant_at_another_group() -> None:
    rng = np.random.default_rng(5)
    x = np.concatenate(_guard_rows(rng, 128)[:2] + [_bf16_rows(rng, 1, 128)[0]])
    q, s = T.ue8m0_group_quant_row(x, G=64)
    assert list(evaluate(bind(DG.Fp8GroupQuantUe8m0, K=384, G=64), _words(x))) == _words(q) + _words(s)


def _weights(rng, *shape) -> np.ndarray:
    w = rng.integers(0, 256, shape).astype(np.uint8)
    w[(w & 0x7F) == 0x7F] ^= 1
    return w


@pytest.mark.parametrize("N", [128, 192])
def test_linears_match_their_twins(N: int) -> None:
    rng = np.random.default_rng(N)
    K = 256
    ws = T.upcast_e8m0(rng.integers(112, 124, (-(-N // 128), K // 128)))
    w = _weights(rng, N, K)
    finite = lambda ws, nf: [v for v, f in zip(ws, nf) if not f]
    guards = _guard_rows(rng, K)
    # (row, every DeepGEMM lane nonfinite, every FlashInfer lane nonfinite): an +inf or a NaN in a group makes every lane NaN (the
    # twin flags it), and so does FlashInfer's all-subnormal group (scale +inf, words 0x7F) where DeepGEMM's eps keeps it finite
    for x, dg_nf, fi_nf in [(_bf16_rows(rng, 1, K)[0], False, False), (guards[1], False, False), (guards[4], True, True),
                            (guards[10], False, True)]:
        q, s, y, nf = T.deepgemm_linear_row(x, w, ws)
        got = list(evaluate(bind(DG.DeepGemmFp8Linear, K=K, N=N), _words(x), _words(w), _words(ws)))
        assert got[:K + K // 128] == _words(q) + _words(s) and finite(got[K + K // 128:], nf) == finite(_words(y), nf)
        assert nf.all() == dg_nf and nf.any() == dg_nf
        y, nf = T.flashinfer_linear_row(x, w, ws)
        got = list(evaluate(bind(DG.FlashInferFp8Linear, K=K, N=N), _words(x), _words(w), _words(ws)))
        assert finite(got, nf) == finite(_words(y), nf) and nf.all() == fi_nf and nf.any() == fi_nf


def test_grouped_einsum_matches_its_twin() -> None:
    rng = np.random.default_rng(6)
    H, K, N, G = 2, 256, 130, 128
    xq, w = _weights(rng, H, K), _weights(rng, H, N, K)
    sx, ws = T.upcast_e8m0(rng.integers(118, 126, (H, K // G))), T.upcast_e8m0(rng.integers(112, 124, (H, 2, K // G)))
    y, _ = T.scaled_mm_fp8_block_grouped_row(xq, sx, w, ws)
    assert list(evaluate(bind(DG.ScaledMmFp8BlockGrouped, H=H, K=K, N=N, G=G), *map(_words, (xq, sx, w, ws)))) == _words(y)


def test_replay_evaluates_the_linears_with_their_row_kernels() -> None:
    """`evaluate_batch`'s default for the three Definitions is their row kernel (`kernels.deepgemm_rows`, registered from `rows`), not
    the plain-integer reference.  The middle row of each batch is one the GEMM twin does not reproduce (DeepGEMM: accumulators past
    the largest finite; FlashInfer: its all-subnormal group's NaN words; the einsum: a NaN e4m3 byte in one token): the row kernel
    declines it by name, and the reference evaluates it."""
    from verity.ml.circuits.evaluation.reference import statics_of
    from verity_vllm.program.kernels import kernel_registry as KR
    from verity_vllm.program.kernels import rows

    rng = np.random.default_rng(8)
    K, N, H, G = 256, 130, 2, 128
    around = lambda mid: np.stack([_bf16_rows(rng, 1, K)[0], mid, _bf16_rows(rng, 1, K)[0]])       # noqa: E731
    shared = lambda *ops: [np.broadcast_to(o, (3, *o.shape)) for o in ops]                         # noqa: E731
    w, ws = _weights(rng, N, K), T.upcast_e8m0(rng.integers(112, 124, (2, K // G)))
    xq, sx = _weights(rng, 3, H, K), T.upcast_e8m0(rng.integers(118, 126, (3, H, K // G)))
    xq[1, 0, 5] = 0x7F
    cases = [(bind(DG.DeepGemmFp8Linear, K=K, N=N), [around(_bf16_rows(rng, 1, K, 1e37)[0]), *shared(w, ws)]),
             (bind(DG.FlashInferFp8Linear, K=K, N=N), [around(_guard_rows(rng, K)[0]), *shared(w, ws)]),
             (bind(DG.ScaledMmFp8BlockGrouped, H=H, K=K, N=N, G=G),
              [xq, sx, *shared(_weights(rng, H, N, K), T.upcast_e8m0(rng.integers(112, 124, (H, 2, K // G))))])]
    for target, ops in cases:
        assert set(E.kernels(target)) == {rows.KERNEL}
        fn, statics = KR.instance(target, rows.KERNEL), statics_of(target)
        with pytest.raises(KR.Declined, match="NaN e4m3 operand or a non-finite accumulator"):
            fn(*(o[1] for o in ops), **statics)
        for i in (0, 2):
            fn(*(o[i] for o in ops), **statics)
        got, ref = E.evaluate_batch(target, *ops), E.evaluate_batch(target, *ops, kernel="reference")
        assert got.dtype == ref.dtype and np.array_equal(got, ref), target.id


def test_statics_are_checked() -> None:
    with pytest.raises(ValueError):
        evaluate(bind(DG.Fp8GroupQuantUe8m0, K=96, G=64), [0] * 96)
    with pytest.raises(ValueError):
        evaluate(bind(DG.FlashInferFp8Quant1x128, K=192), [0] * 192)


# --- on bits ---------------------------------------------------------------------------------------------------------------------

AGREEMENT = [bind(BDG.F32LtFtzV2), bind(BDG.Fp8Ue8m0CeilV2), bind(BDG.Fp8GroupScaleUe8m0V2, G=32),
             bind(BDG.Fp8GroupQuantUe8m0GroupV2, G=32), bind(BDG.Fp8GroupQuantUe8m0V2, K=64, G=32), bind(BDG.FlashInferFp8Quant1x128GroupV2),
             bind(BDG.FlashInferFp8Quant1x128V2, K=256), bind(BDG.DeepGemmFp8LinearV2, K=128, N=2),
             bind(BDG.FlashInferFp8LinearV2, K=128, N=2), bind(BDG.ScaledMmFp8BlockGroupedV2, H=2, K=64, N=34, G=32)]


@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_boolean_versions_agree_with_their_word_views(target) -> None:
    word = TR.word_of(target)
    assert BL.is_boolean(target) and word.id == target.id.replace("_v2", "_v1", 1)
    assert SW.boolean_version(word).id == target.id
    n = 64 if BS.compile(target).counts["and"] > 200_000 else 400
    inputs = _inputs(word, n, sum(map(ord, target.id)))
    got, ref = _flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))
    for g, r in zip(got, ref, strict=True):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"


def test_flashinfer_group_on_bits_at_its_guards() -> None:
    rng = np.random.default_rng(7)
    rows = np.stack(_guard_rows(rng, 128))
    target = bind(BDG.FlashInferFp8Quant1x128GroupV2)
    got, ref = _flat(BS.evaluate_words(target, rows)), _flat(E.evaluate_batch(TR.word_of(target), rows))
    assert all((g == r).all() for g, r in zip(got, ref, strict=True))


# --- the load-time weight processing --------------------------------------------------------------------------------------------

def test_e8m0_upcast_is_vllms() -> None:
    torch = pytest.importorskip("torch")
    b = torch.arange(256, dtype=torch.uint8)
    want = (b.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)            # fp8_utils.py:1071-1080
    assert (T.upcast_e8m0(b.numpy()) == want.numpy().view(np.uint32)).all()
    assert _words(T.upcast_e8m0(np.array([0, 127, 255])))== [0, ONE, INF]


def _vllm_per_block_cast_to_fp8(x, block):
    """vllm/utils/deep_gemm.py:720-740 at d9105ea80, use_ue8m0=True, run eagerly (its body, without `torch.compile`)."""
    import torch
    m, n = x.shape
    bm, bn = block
    xp = torch.zeros((-(-m // bm) * bm, -(-n // bn) * bn), dtype=x.dtype)
    xp[:m, :n] = x
    xv = xp.view(-1, bm, xp.size(1) // bn, bn)
    amax = xv.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = amax / 448.0
    sf = torch.pow(2.0, torch.ceil(torch.log2(sf.abs())))
    q = (xv * (1.0 / sf)).to(torch.float8_e4m3fn)
    return q.view_as(xp)[:m, :n].contiguous(), sf.view(xv.size(0), xv.size(2))


def test_ue8m0_requant_against_vllms_cast() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(8)
    N, K = 320, 320                                                    # a partial last block in both dimensions
    w = _weights(rng, N, K)
    s = R.as_u32((np.abs(rng.standard_normal((3, 3))) * 1e-3 + 1e-5).astype(np.float32))
    s[0, 1] = 0x3A800000                                               # 2^-10: amax / 448 lands on a power of two
    w[0:128, 128:256][0, 0] = 0x7E                                     # with 448 in its block
    q, sf, unresolved = T.requant_weight_ue8m0(w, s)
    s_exp = np.ascontiguousarray(np.repeat(np.repeat(R.as_f32(s), 128, 0), 128, 1)[:N, :K])
    w_dq = torch.from_numpy(w).view(torch.float8_e4m3fn).float() * torch.from_numpy(s_exp)
    wq, ws = _vllm_per_block_cast_to_fp8(w_dq, (128, 128))
    assert unresolved[0, 1] and unresolved.sum() == 1
    ok = np.repeat(np.repeat(~unresolved, 128, 0), 128, 1)[:N, :K]
    assert (wq.view(torch.uint8).numpy()[ok] == q[ok]).all()
    assert (ws.numpy().view(np.uint32)[~unresolved] == sf[~unresolved]).all()
    assert int(sf[0, 1]) == 0x3A800000
