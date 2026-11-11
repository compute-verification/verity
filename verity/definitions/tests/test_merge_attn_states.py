"""`registry.merge_attn_states`: the Definition is word for word the CPU twin of vLLM d9105ea80's
`merge_attn_states_kernel<__nv_bfloat16, __nv_bfloat16, 128, false>` read from its sm_90 SASS (`kernel_twin`), on edge words and at
GLM-5.3's served statics; a token without context is its suffix's words; the merge is the softmax merge within bf16 rounding; a
head size the kernel can't take is refused."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation.reference import evaluate
from verity.definitions.numerics import derived_rows as DR
from verity.definitions.numerics.moe_twins import nv_expf_bits
from verity.definitions.registry import merge_attn_states as M

U32 = np.uint32
NEG_INF, POS_INF = U32(0xFF800000), U32(0x7F800000)


def kernel_sums(p_lse, s_lse):
    """The SASS up to the weights' sum, on f32 word arrays: (whether the max is infinite, p_se, s_se, tot).  FSETP.NEU |x|, +INF /
    FSEL (an infinite lse is -inf), FMNMX (fmaxf), FADD of the negated max, `__nv_expf`, FADD."""
    def neg_inf_if_inf(x):
        x = np.asarray(x, U32)
        return np.where((x & U32(0x7FFFFFFF)) == POS_INF, NEG_INF, x)

    pl, sl = neg_inf_if_inf(p_lse), neg_inf_if_inf(s_lse)
    m = DR.as_u32(np.fmax(DR.as_f32(pl), DR.as_f32(sl)))
    expf = np.vectorize(nv_expf_bits, otypes=[U32])
    p_se = expf(DR.add_rn(pl, m ^ U32(0x80000000)))
    s_se = expf(DR.add_rn(sl, m ^ U32(0x80000000)))
    return (m & U32(0x7FFFFFFF)) == POS_INF, p_se, s_se, DR.add_rn(p_se, s_se)


def _f32(w, weight):
    """bf16 words [..., D] widened, and a weight [...] made to broadcast against them."""
    return DR.bf16_to_f32_words(w), np.asarray(weight, U32)[..., None]


def kernel_merge(p, s, p_se, s_se, tot):
    """The two div.rn, FMUL of the suffix's term, FFMA of the prefix's: the f32 words F2FP rounds."""
    (pf, ps), (sf, ss) = _f32(p, DR.div_rn(p_se, tot)), _f32(s, DR.div_rn(s_se, tot))
    return DR.fma_rn(pf, ps, DR.mul_rn(sf, ss))


def kernel_twin(p, p_lse, s, s_lse):
    """The kernel's SASS on word arrays, independent of the Definition: p, s [..., NH, D] bf16 words, p_lse, s_lse [..., NH] f32
    words -> [..., NH, D] bf16 words; the prefix's words where the max is infinite, else F2FP of `kernel_merge`."""
    empty, p_se, s_se, tot = kernel_sums(p_lse, s_lse)
    return np.where(empty[..., None], np.asarray(p, np.uint16), DR.f2fp_bf16(kernel_merge(p, s, p_se, s_se, tot)))


def _fmul_fadd(p, s, p_se, s_se, tot):
    (pf, ps), (sf, ss) = _f32(p, DR.div_rn(p_se, tot)), _f32(s, DR.div_rn(s_se, tot))
    return DR.add_rn(DR.mul_rn(pf, ps), DR.mul_rn(sf, ss))


def _ffma_of_the_suffix(p, s, p_se, s_se, tot):
    (pf, ps), (sf, ss) = _f32(p, DR.div_rn(p_se, tot)), _f32(s, DR.div_rn(s_se, tot))
    return DR.fma_rn(sf, ss, DR.mul_rn(pf, ps))


def _reciprocal(p, s, p_se, s_se, tot):
    r = DR.div_rn(np.full_like(tot, 0x3F800000), tot)
    (pf, ps), (sf, ss) = _f32(p, DR.mul_rn(p_se, r)), _f32(s, DR.mul_rn(s_se, r))
    return DR.fma_rn(pf, ps, DR.mul_rn(sf, ss))


#: readings of the SASS one rounding off the kernel's, which `kernel_merge`'s order rules out: unfused p*ps + s*ss, the FFMA on the
#: suffix's term, and one reciprocal of the sum for both weights
MISREADINGS = {"fmul-fadd": _fmul_fadd, "ffma-of-the-suffix": _ffma_of_the_suffix, "reciprocal": _reciprocal}


def _definition(p, p_lse, s, s_lse):
    """`MergeAttnStates_v1{NH, D}` evaluated per token (axis 0)."""
    n, NH, D = p.shape
    fn = bind(M.MergeAttnStates, NH=NH, D=D)
    rows = [evaluate(fn, p[t].reshape(-1).tolist(), p_lse[t].tolist(), s[t].reshape(-1).tolist(), s_lse[t].tolist()) for t in range(n)]
    return np.array(rows, np.uint16).reshape(n, NH, D)


def _bf16(rng, shape, sd=1.0):
    return (rng.normal(0.0, sd, shape).astype(np.float32).view(U32) >> 16).astype(np.uint16)


def _f32_words(x):
    return np.asarray(x, np.float32).view(U32)


#: lse words the kernel meets or must survive: a no-context prefix (-inf), +inf (a backend's empty row), NaN, signed zeros,
#: subnormals, and the extremes where one weight underflows
EDGE_LSE = [0xFF800000, 0x7F800000, 0x7FC00000, 0xFFFFFFFF, 0x00000000, 0x80000000, 0x00000001, 0x807FFFFF,
            *_f32_words([88.5, -88.5, 104.0, -104.0, 1e30, -1e30, 3.0, 3.0000002]).tolist()]
#: output words beside the random ones: zeros, the smallest subnormal, the largest finite, the infinities, a NaN
EDGE_OUT = [0x0000, 0x8000, 0x0001, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC1]


def test_definition_is_the_kernel_twin_on_edge_words() -> None:
    """Every pair of edge lse words, on random outputs with the edge words planted in them."""
    rng = np.random.default_rng(1)
    pairs = [(a, b) for a in EDGE_LSE for b in EDGE_LSE]
    n, NH, D = len(pairs), 1, 16
    p_lse = np.array([[a] for a, _ in pairs], U32)
    s_lse = np.array([[b] for _, b in pairs], U32)
    p, s = _bf16(rng, (n, NH, D)), _bf16(rng, (n, NH, D))
    p[:, 0, :len(EDGE_OUT)] = EDGE_OUT
    s[:, 0, D - len(EDGE_OUT):] = EDGE_OUT
    np.testing.assert_array_equal(_definition(p, p_lse, s, s_lse), kernel_twin(p, p_lse, s, s_lse))


def test_definition_is_the_kernel_twin_on_random_tokens() -> None:
    """Random tokens of two heads: the lse pairs spread from equal to far apart."""
    rng = np.random.default_rng(2)
    n, NH, D = 48, 2, 16
    p_lse = _f32_words(rng.normal(0.0, 8.0, (n, NH)))
    s_lse = _f32_words(rng.normal(0.0, 8.0, (n, NH)))
    p, s = _bf16(rng, (n, NH, D), 0.5), _bf16(rng, (n, NH, D), 0.5)
    np.testing.assert_array_equal(_definition(p, p_lse, s, s_lse), kernel_twin(p, p_lse, s, s_lse))


def test_definition_is_the_kernel_twin_where_one_rounding_shows() -> None:
    """Heads whose elements are the ones where a reading in MISREADINGS gives another bf16 word than the kernel, found through the
    twin, at least two words for each reading: random elements show such a word a few times in 10^5, so only these hold the
    Definition to the kernel's order of roundings."""
    rng = np.random.default_rng(6)
    D, POOL, HEADS = 8, 1 << 16, 96
    hits = dict.fromkeys(MISREADINGS, 0)
    rows = []
    for _ in range(HEADS):
        if min(hits.values()) >= 2:
            break
        pl, sl = _f32_words(rng.normal(0.0, 2.0, (2, 1)))
        _, p_se, s_se, tot = kernel_sums(pl, sl)
        cp, cs = _bf16(rng, (1, POOL)), _bf16(rng, (1, POOL))
        want = DR.f2fp_bf16(kernel_merge(cp, cs, p_se, s_se, tot))[0]
        other = {k: DR.f2fp_bf16(f(cp, cs, p_se, s_se, tot))[0] != want for k, f in MISREADINGS.items()}
        shown = np.flatnonzero(np.logical_or.reduce(list(other.values())))[:D]
        if shown.size:
            pick = np.concatenate([shown, np.setdiff1d(np.arange(D), shown)[:D - shown.size]])
            rows.append((pl[0], sl[0], cp[0, pick], cs[0, pick]))
            for k in hits:
                hits[k] += int(other[k][shown].sum())
    assert min(hits.values()) >= 2, hits
    p_lse, s_lse = (np.array([[r[i]] for r in rows], U32) for i in (0, 1))
    p, s = (np.array([[r[i]] for r in rows], np.uint16) for i in (2, 3))
    np.testing.assert_array_equal(_definition(p, p_lse, s, s_lse), kernel_twin(p, p_lse, s, s_lse))


def test_served_statics_one_token() -> None:
    """GLM-5.3 TP8's dense-MHA prefill: one token's 8 heads of 256 (`forward_mha`'s v_head_dim), one head without context."""
    rng = np.random.default_rng(3)
    NH, D = 8, 256
    p_lse = _f32_words(rng.normal(2.0, 3.0, (1, NH)))
    p_lse[0, 5] = NEG_INF
    s_lse = _f32_words(rng.normal(2.0, 3.0, (1, NH)))
    p, s = _bf16(rng, (1, NH, D), 0.3), _bf16(rng, (1, NH, D), 0.3)
    p[0, 5] = 0
    got = _definition(p, p_lse, s, s_lse)
    np.testing.assert_array_equal(got, kernel_twin(p, p_lse, s, s_lse))
    np.testing.assert_array_equal(got[0, 5], s[0, 5])


def test_a_token_without_context_is_its_suffix() -> None:
    """`init_mla_context_partial`'s zero prefix with lse -inf: ps = +0 and ss = 1, so every suffix word comes back as it is, but -0,
    which comes back +0 (+0 + -0 rounds to +0), and a NaN, which comes back 0x7FFF."""
    rng = np.random.default_rng(4)
    n, NH, D = 8, 2, 8
    s = _bf16(rng, (n, NH, D), 4.0)
    s[0, 0] = [0x8000, 0x0000, 0x0001, 0x8001, 0x7F7F, 0xFF80, 0x7FC1, 0xFFC0]
    s_lse = _f32_words(rng.normal(0.0, 20.0, (n, NH)))
    got = _definition(np.zeros_like(s), np.full((n, NH), NEG_INF, U32), s, s_lse)
    want = s.copy()
    want[0, 0] = [0x0000, 0x0000, 0x0001, 0x8001, 0x7F7F, 0xFF80, 0x7FFF, 0x7FFF]
    np.testing.assert_array_equal(got, want)


def test_merge_is_the_softmax_merge_within_bf16_rounding() -> None:
    """Against the real-valued merge (e^p_lse p + e^s_lse s) / (e^p_lse + e^s_lse): within one bf16 rounding and the f32 weights'."""
    rng = np.random.default_rng(5)
    n, NH, D = 24, 2, 16
    pl, sl = rng.normal(0.0, 2.0, (n, NH)), rng.normal(0.0, 2.0, (n, NH))
    p, s = _bf16(rng, (n, NH, D)), _bf16(rng, (n, NH, D))
    got = DR.as_f32(DR.bf16_to_f32_words(_definition(p, _f32_words(pl), s, _f32_words(sl)))).astype(np.float64)
    pf, sf = (DR.as_f32(DR.bf16_to_f32_words(x)).astype(np.float64) for x in (p, s))
    pl32, sl32 = np.float32(pl).astype(np.float64), np.float32(sl).astype(np.float64)
    m = np.maximum(pl32, sl32)
    wp, ws = np.exp(pl32 - m), np.exp(sl32 - m)
    want = (wp[..., None] * pf + ws[..., None] * sf) / (wp + ws)[..., None]
    scale = np.maximum(np.abs(pf), np.abs(sf))
    assert np.all(np.abs(got - want) <= 2.0 ** -8 * np.abs(want) + 2.0 ** -20 * scale)


@pytest.mark.parametrize("F,statics,why", [(M.MergeAttnStates, {"NH": 1, "D": 12}, "multiple of 8"),
                                           (M.MergeAttnStatesHead, {"D": 4}, "multiple of 8"),
                                           (M.MergeAttnStates, {"NH": 0, "D": 8}, "at least one head")])
def test_refuses_a_binding_the_kernel_cannot_take(F, statics, why) -> None:
    with pytest.raises(ValueError, match=why):
        bind(F, **statics)
